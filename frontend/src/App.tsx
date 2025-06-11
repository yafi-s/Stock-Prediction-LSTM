import React from 'react';
import { Box, CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import StockPrediction from './StockPrediction';
import StarryBackground from './components/StarryBackground';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    background: {
      default: 'transparent',
      paper: 'rgba(0, 0, 0, 0.7)',
    },
  },
});

const App: React.FC = () => {
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box
        sx={{
          minHeight: '100vh',
          width: '100vw',
          overflow: 'hidden',
          position: 'relative',
          background: 'linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%)',
          '& #tsparticles': {
            position: 'absolute',
            width: '100%',
            height: '100%',
            zIndex: 0
          }
        }}
      >
        <StarryBackground />
        <Box
          sx={{
            position: 'relative',
            zIndex: 1,
            padding: 3,
            minHeight: '100vh',
            '& canvas': {
              display: 'block',
              verticalAlign: 'bottom',
            }
          }}
        >
          <StockPrediction />
        </Box>
      </Box>
    </ThemeProvider>
  );
};

export default App;
