import React, { useCallback, useEffect, useState } from 'react';
import Particles from 'react-tsparticles';
import { loadFull } from 'tsparticles';

const StarryBackground: React.FC = () => {
  const [parallaxOffset, setParallaxOffset] = useState(0);

  const particlesInit = useCallback(async (engine: any) => {
    await loadFull(engine);
  }, []);

  useEffect(() => {
    let animationFrameId: number;
    let currentOffset = parallaxOffset;

    const handleScroll = () => {
      const scrollPosition = window.scrollY;
      
      // Calculate target parallax offset with easing
      const targetOffset = scrollPosition * 0.3;
      
      // Smooth animation function
      const animate = () => {
        // Easing function for smooth movement
        const easing = 0.1;
        currentOffset += (targetOffset - currentOffset) * easing;
        
        setParallaxOffset(currentOffset);
        
        // Continue animation if we're still scrolling
        if (Math.abs(targetOffset - currentOffset) > 0.1) {
          animationFrameId = requestAnimationFrame(animate);
        }
      };

      // Start animation
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
      animationFrameId = requestAnimationFrame(animate);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => {
      window.removeEventListener('scroll', handleScroll);
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [parallaxOffset]);

  return (
    <div 
      style={{ 
        position: 'fixed',
        top: '-50%',
        left: 0,
        width: '100%',
        height: '200%',
        transform: `translateY(${parallaxOffset}px)`,
        willChange: 'transform',
        pointerEvents: 'none',
        overflow: 'hidden'
      }}
    >
      <Particles
        id="tsparticles"
        init={particlesInit}
        options={{
          fullScreen: {
            enable: true,
            zIndex: 0
          },
          background: {
            color: {
              value: "transparent"
            }
          },
          particles: {
            number: {
              value: 300,
              density: {
                enable: true,
                value_area: 800
              }
            },
            color: {
              value: "#ffffff"
            },
            shape: {
              type: "circle"
            },
            opacity: {
              value: 0.65,
              random: true,
              animation: {
                enable: true,
                speed: 1,
                minimumValue: 0.3,
                sync: false
              }
            },
            size: {
              value: 1.5,
              random: true,
              animation: {
                enable: true,
                speed: 2,
                minimumValue: 1,
                sync: false
              }
            },
            move: {
              enable: true,
              speed: 0.5,
              direction: "none",
              random: true,
              straight: false,
              outModes: {
                default: "out"
              }
            }
          },
          detectRetina: false,
          fpsLimit: 30
        }}
      />
    </div>
  );
};

export default StarryBackground; 