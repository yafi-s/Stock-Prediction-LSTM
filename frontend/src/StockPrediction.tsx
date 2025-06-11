// src/components/StockPrediction.tsx

import React, { useState, useMemo, useEffect, useRef } from "react";
import type { ReactElement } from "react";

import {
  Box,
  Typography,
  TextField,
  Button,
  Table,
  TableHead,
  TableBody,
  TableCell,
  TableRow,
  Paper,
} from "@mui/material";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";


interface HistoricalPoint {
  date: string;  // e.g. "2025-06-01"
  price: number; // e.g. 202.00
}

interface Prediction {
  date: string;               // e.g. "2025-06-05"
  predictedPrice: number;     // e.g. 202.81
  confidenceInterval: number; // e.g. 4.06
}

interface ApiResponse {
  symbol: string;
  lastActualClose: number;
  lastUpdated: string;
  mape: number | null;
  predictions: {
    dates: string[];
    predictions: number[];
    confidenceIntervals: number[];
  };
  error?: boolean;
  message?: string;
}

interface ChartPoint {
  date: string;
  price: number;
  isPredicted: boolean;
}

const StockPrediction: React.FC = () => {
  const [symbol, setSymbol] = useState<string>("AAPL");
  const [displayedSymbol, setDisplayedSymbol] = useState<string>("AAPL");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [lastClose, setLastClose] = useState<number | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string>("");
  const [history, setHistory] = useState<HistoricalPoint[]>([]);
  const [predictions, setPredictions] = useState<Prediction[]>([]);

  // State to control axes opacity for fade-in animation
  const [lineOpacity, setLineOpacity] = useState<number>(0);
  const [axesOpacity, setAxesOpacity] = useState<number>(0);

  // Ref to store the line length
  const lineLengthRef = useRef<number | null>(null);

  const [scalingFactor, setScalingFactor] = useState<number>(1);
  const [meanPrice, setMeanPrice] = useState<number>(0);

  const [chartOpacity, setChartOpacity] = useState<number>(1);

  // Function to scale data
  const scaleData = (data: any[]) => {
    if (data.length === 0) return data;
    
    const prices = data.map(d => d.price);
    const mean = prices.reduce((a, b) => a + b, 0) / prices.length;
    const max = Math.max(...prices);
    const min = Math.min(...prices);
    const scale = max - min;
    
    setMeanPrice(mean);
    setScalingFactor(scale);
    
    return data.map(d => ({
      ...d,
      price: (d.price - mean) / scale
    }));
  };

  // Function to unscale data
  const unscaleData = (data: any[]) => {
    return data.map(d => ({
      ...d,
      price: d.price * scalingFactor + meanPrice
    }));
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // ─────────────────────────────────────────────────────────────────────────────
  useEffect(() => {
    document.body.style.backgroundColor = "#000208";

  }, []); // Empty dependency array ensures this runs only once on mount

  // ─────────────────────────────────────────────────────────────────────────────
  // 3) Fetch function: GET `/api/predict/:symbol`
  // ─────────────────────────────────────────────────────────────────────────────
  const fetchPredictions = async (ticker: string): Promise<void> => {
    try {
      setIsLoading(true);
      setError(null);

      // Start fade out sequence
      const fadeOut = async () => {
        setChartOpacity(0);
        setAxesOpacity(0);
        await new Promise(resolve => setTimeout(resolve, 500));
      };

      await fadeOut();

      const response: Response = await fetch(`/api/predict/${ticker}`);

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Server returned status ${response.status}: ${errorText}`);
        throw new Error(`Server returned status ${response.status}: ${errorText}`);
      }

      const data: ApiResponse = await response.json();

      // Check for error in response
      if (data.error) {
        setError(data.message || "An error occurred");
        setPredictions([]);
        setLastClose(null);
        setLastUpdated("");
        return;
      }

      const transformedHistory: HistoricalPoint[] = [];

      const transformedPredictions: Prediction[] = data.predictions.dates.map((date, index) => ({
        date: date,
        predictedPrice: data.predictions.predictions[index],
        confidenceInterval: data.predictions.confidenceIntervals[index],
      }));

      setLastClose(data.lastActualClose);
      setLastUpdated(data.lastUpdated);
      setHistory(transformedHistory);
      setPredictions(transformedPredictions);
      setDisplayedSymbol(ticker);

      // Scale the data before setting state
      const scaledData = scaleData(transformedHistory);
      const scaledPredictions = transformedPredictions.map((p) => ({
        date: p.date,
        price: (p.predictedPrice - meanPrice) / scalingFactor,
        isPredicted: true,
      }));

      // Wait a brief moment after data is loaded before starting fade in
      await new Promise(resolve => setTimeout(resolve, 100));

      // Fade in sequence - keep the sequential fade in
      const fadeIn = async () => {
        // First fade in the chart
        setChartOpacity(1);
        // Wait for chart fade in
        await new Promise(resolve => setTimeout(resolve, 500));
        // Then fade in the axes
        setAxesOpacity(1);
      };

      // Execute fade in sequence
      await fadeIn();

    } catch (err: unknown) {
      console.error("Error during fetchPredictions:", err);
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Unknown error");
      }
    } finally {
      setIsLoading(false);
    }
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // 4) Build `chartData`
  // ─────────────────────────────────────────────────────────────────────────────
  const chartData: ChartPoint[] = useMemo(() => {
    if (predictions.length === 0) {
      return [];
    }

    const sortedPreds: Prediction[] = [...predictions].sort(
      (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
    );

    const predPoints: ChartPoint[] = sortedPreds.map((pt) => ({
      date: pt.date,
      price: pt.predictedPrice,
      isPredicted: true,
    }));

    return predPoints;
  }, [predictions]);

  // ─────────────────────────────────────────────────────────────────────────────
  // Animation effect for the chart line and axes fade-in
  // ─────────────────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (chartData && chartData.length > 0) {
      // Reset opacities when data changes
      setLineOpacity(0);
      setAxesOpacity(0);

      // Fade in the line after 1.5 seconds
      const lineTimer = setTimeout(() => {
        setLineOpacity(1);
      }, 1500);

      // Fade in the axes after 2.5 seconds
      const axesTimer = setTimeout(() => {
        setAxesOpacity(1);
      }, 2500);

      // Cleanup timers
      return () => {
        clearTimeout(lineTimer);
        clearTimeout(axesTimer);
      };
    }
  }, [chartData]);

  // Callback ref for the chart line animation
  const setChartLineRef = React.useCallback((node: any) => {
    // Ensure node is the SVGPathElement rendered by Recharts
    if (node) {
      const pathElement = node.tagName === 'path' ? node : (node._node && node._node.tagName === 'path' ? node._node : null);

      if (pathElement) {
        const totalLength = pathElement.getTotalLength();

        // Set the initial strokeDasharray and strokeDashoffset
        pathElement.style.strokeDasharray = `${totalLength}`;
        pathElement.style.strokeDashoffset = `${totalLength}`;

        // Store the length to use in the useSpring hook
        lineLengthRef.current = totalLength;

        // Trigger the animation by updating the spring value
        setTimeout(() => {
          setLineOpacity(1);
        }, 0);
      }
    }
  }, [chartData]); // Dependency on chartData to re-run animation when data changes

  // ─────────────────────────────────────────────────────────────────────────────
  // 5) Custom dot renderer:
  // ─────────────────────────────────────────────────────────────────────────────
  const renderDot = (props: any): ReactElement<SVGElement> => {
    const { cx, cy, payload } = props;
    const isPrediction = payload.isPredicted;
    
    return (
      <circle
        cx={cx}
        cy={cy}
        r={isPrediction ? 5 : 3}
        fill="#ffffff"
        stroke="#ffffff"
        strokeWidth={1}
        filter="url(#dotGlow)"
        style={{ 
          opacity: chartOpacity,
          transition: 'opacity 1s ease-in-out'
        }}
      />
    );
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // 6) Helper: Format "YYYY-MM-DD" → "M/DD"
  // ─────────────────────────────────────────────────────────────────────────────
  const formatAsMD = (dateString: string): string => {
    const d = new Date(dateString);
    const m = d.getMonth() + 1; // 1–12
    const day = d.getDate();    // 1–31
    const dd = day < 10 ? `0${day}` : `${day}`;
    return `${m}/${dd}`; // e.g. "6/05"
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // 7) JSX return (TSX)
  // ─────────────────────────────────────────────────────────────────────────────
  return (
    <> {/* Use a fragment as the top-level element */}
      <Typography variant="h3" gutterBottom align="center" sx={{ mt: 4, mb: 6, width: '100%', fontWeight: 'bold' }}>
        Stock Price Prediction
      </Typography>

      {/* SYMBOL INPUT + BUTTON - Moved outside main content box and centered */}
      {/* Outer Box for centering */}
      <Box
        sx={{
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          width: '100%', 
          mb: 5, 
        }}
      >
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 2,
            padding: '12px', 
            borderRadius: '4px', 
            margin: 'auto', 
          }}
        >
          <TextField
            label="Stock Symbol"
            variant="outlined"
            size="medium"
            value={symbol}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
              setSymbol(e.target.value.toUpperCase())
            }
            sx={{
              width: 200,
              '& .MuiOutlinedInput-root': {
                fieldset: { borderColor: '#888' }, // Adjust border color to lighter gray
                '&:hover fieldset': { borderColor: '#aaa' }, // Adjust hover border color
                '&.Mui-focused fieldset': { borderColor: '#90caf9' }, // Keep focused color
                color: '#ffffff', // Text color
              },
              '& .MuiInputLabel-root': {
                color: '#bbb', // Label color
                '&.Mui-focused': { color: '#90caf9' }, // Focused label color
              },
            }}
          />
          <Button
            variant="contained"
            sx={{
              px: 4,
              py: '14.5px', 
              backgroundColor: '#cccccc', 
              '&:hover': { backgroundColor: '#e0e0e0' }, 
              color: '#000000', 
              boxShadow: 'none',
            }}
            onClick={() => fetchPredictions(symbol)}
            disabled={isLoading}
          >
            {isLoading ? "Loading…" : "PREDICT"}
          </Button>
        </Box> {/* Close inner Box */}
      </Box> {/* Close outer Box */}

      {/* Main content area to span full width with padding - now contains messages, chart, and table */}
      <Box
        sx={{
          width: '100%', 
          maxWidth: '1200px', 
          mx: 'auto', 
          px: { xs: 2, sm: 4, md: 6 }, 
          boxSizing: 'border-box', 
          flexShrink: 0, 
          borderRadius: '4px', 
          py: 4, 
        }}
      >

        {/* ERROR TEXT */}
        {error && (
          <Typography color="error" sx={{ mb: 3, fontSize: "1rem", fontWeight: 'bold' }}>
            Error: {error}
          </Typography>
        )}

        {/* PROMPT WHEN NO DATA */}
        {!predictions.length && !error && !isLoading && (
          <Typography color="text.secondary" sx={{ fontSize: "1.1rem", fontStyle: 'italic' }}>
            Enter a symbol and click "PREDICT" to see results.
          </Typography>
        )}

        {/* LOADING INDICATOR */}
        {isLoading && !predictions.length && !error && (
            <Typography color="text.secondary" sx={{ fontSize: "1.1rem", fontStyle: 'italic' }}>
                Loading predictions...
            </Typography>
        )}

        {/* CHART + TABLE (only if there's data) */}
        {predictions.length > 0 && lastClose !== null && (
          <>
            {/* ─── RECHARTS LINE CHART ─────────────────────────────────── */}
            {/* Box for height and to contain the responsive chart, now spanning full width */} 
            <Box
              sx={{
                width: '100vw',
                position: 'relative',
                left: '50%',
                right: '50%',
                marginLeft: '-50vw',
                marginRight: '-50vw',
                height: 500,
                mt: 5,
                mb: 5,
                px: { xs: 2, sm: 4, md: 6 },
                boxSizing: 'border-box',
                opacity: chartOpacity,
                transition: 'all 0.5s ease-in-out',
                visibility: chartOpacity === 0 ? 'hidden' : 'visible'
              }}
            >
              <ResponsiveContainer
                width="100%"
                height="100%"
              >
                <LineChart
                  data={chartData}
                  margin={{ top: 30, right: 40, left: 0, bottom: 0 }}
                >
                  <defs>
                    <filter id="glow">
                      <feGaussianBlur stdDeviation="5" result="coloredBlur"/>
                      <feMerge>
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                      </feMerge>
                    </filter>
                    <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#8884d8"/>
                      <stop offset="100%" stopColor="#502bd4"/>
                    </linearGradient>
                    <filter id="dotGlow" x="-50%" y="-50%" width="200%" height="200%">
                      <feGaussianBlur stdDeviation="2" result="blur" />
                      <feComposite in="SourceGraphic" in2="blur" operator="over" />
                    </filter>
                  </defs>
                  <XAxis
                    dataKey="date"
                    tickFormatter={(dateStr: string) => formatAsMD(dateStr)}
                    tick={{ fill: "#ffffff", fontSize: 12 }}
                    axisLine={{ stroke: "#888", strokeWidth: 1 }}
                    tickLine={false}
                    interval={0}
                    padding={{ left: 20, right: 20 }}
                    angle={-30}
                    textAnchor="end"
                    style={{ 
                      opacity: axesOpacity, 
                      transition: 'all 0.5s ease-in-out',
                      visibility: axesOpacity === 0 ? 'hidden' : 'visible'
                    }}
                  />
                  <YAxis
                    tickFormatter={(val) => (val as number).toFixed(2)}
                    tick={{ fill: "#ffffff", fontSize: 12 }}
                    axisLine={{ stroke: "#888", strokeWidth: 1 }}
                    tickLine={false}
                    allowDecimals={true}
                    domain={['dataMin', 'dataMax']}
                    style={{ 
                      opacity: axesOpacity, 
                      transition: 'all 0.5s ease-in-out',
                      visibility: axesOpacity === 0 ? 'hidden' : 'visible'
                    }}
                  />
                  <Tooltip
                    wrapperStyle={{ outline: "none" }}
                    contentStyle={{
                      backgroundColor: "#1e1e1e",
                      border: "1px solid #444",
                      fontSize: 14,
                      borderRadius: '4px',
                      padding: '8px',
                      boxShadow: '0px 2px 5px rgba(0,0,0,0.2)',
                      color: "#ffffff",
                    }}
                    labelFormatter={(label: string) => `Date: ${label}`}
                    formatter={(value: any) => [
                      `$${(value * scalingFactor + meanPrice).toFixed(2)}`,
                      "Price",
                    ]}
                  />
                  <Line
                    type="monotone"
                    dataKey="price"
                    stroke="url(#lineGradient)"
                    strokeWidth={4}
                    dot={renderDot}
                    activeDot={{ r: 7, strokeWidth: 2, stroke: "#ffffff" }}
                    isAnimationActive={false}
                    filter="url(#glow)"
                    style={{ 
                      opacity: chartOpacity,
                      transition: 'all 0.5s ease-in-out'
                    }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Box>

            {/* ─── TABLE BELOW THE CHART ─────────────────────────────────── */}
            <Box sx={{ mt: 5, mb: 6 }}>
              <Typography variant="h5" gutterBottom>
                Predictions for {displayedSymbol.toUpperCase()}
              </Typography>
              <Typography sx={{ mb: 3, fontSize: "0.9rem", color: "text.secondary" }}>
                <strong>Last Close:</strong> ${lastClose.toFixed(2)} &nbsp;
                <strong>Last Updated:</strong> {lastUpdated}
              </Typography>

              {/* Wrap Table with Paper for styling,*/}
              <Paper
                variant="outlined"
                sx={{
                  mt: 1,
                  borderColor: '#03011D', // Set outline border color
                  borderRadius: '4px',
                  overflow: 'hidden',
                  boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.3)', 
                }}
              >
                <Table
                  size="medium"
                >
                <TableHead>
                    <TableRow
                      sx={{
                        '& th': {
                          backgroundColor: '#3D3475',
                          color: '#ffffff', // 
                          borderBottom: '1px solid #03011D', // 
                        },
                      }}
                    >
                      <TableCell sx={{ fontSize: 14, fontWeight: 'bold' }}>Date</TableCell>
                      <TableCell align="right" sx={{ fontSize: 14, fontWeight: 'bold' }}>Predicted Price</TableCell>
                      <TableCell align="right" sx={{ fontSize: 14, fontWeight: 'bold' }}>Confidence Interval</TableCell>
                  </TableRow>
                </TableHead>
                  <TableBody
                    sx={{
                    }}
                  >
                    {predictions.map((pred, index) => (
                      <TableRow
                        key={pred.date} 
                        sx={{
                          '&:last-child td, &:last-child th': { border: 0 },
                          '&:hover': {
                            backgroundColor: '#333',
                          },
                          transition: 'background-color 0.3s ease',
                          '& td': {
                            backgroundColor: '#322A5F',
                          },
                        }}
                      >
                        <TableCell
                          component="th"
                          scope="row"
                          sx={{
                            fontSize: 14,
                            color: '#ffffff',
                            backgroundColor: '#3D3475 !important',
                            ...(index < predictions.length - 1 && { borderBottom: '1px solid #03011D' }),
                          }}
                        >
                        {pred.date}
                      </TableCell>
                        <TableCell align="right" sx={{
                          ...(index < predictions.length - 1 && { borderBottom: '1px solid #03011D' }),
                        }}>
                        ${pred.predictedPrice.toFixed(2)}
                      </TableCell>
                        <TableCell align="right" sx={{
                          ...(index < predictions.length - 1 && { borderBottom: '1px solid #03011D' }),
                        }}>
                        ±${pred.confidenceInterval.toFixed(2)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              </Paper>
            </Box>
          </>
        )}
      </Box>
    </> // Close fragment
  );
};

export { StockPrediction as default };
