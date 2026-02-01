'use client';

import { useEffect, useRef, useState } from 'react';

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  color: string;
  opacity: number;
  connections: number[];
}

interface MoleculeBackgroundProps {
  particleCount?: number;
  connectionDistance?: number;
  speed?: number;
  className?: string;
}

const COLORS = [
  'rgba(59, 130, 246, 0.8)',   // Blue
  'rgba(139, 92, 246, 0.8)',   // Purple
  'rgba(6, 182, 212, 0.7)',    // Cyan
  'rgba(16, 185, 129, 0.6)',   // Emerald
  'rgba(244, 114, 182, 0.6)',  // Pink
];

export function MoleculeBackground({
  particleCount = 50,
  connectionDistance = 150,
  speed = 0.5,
  className = '',
}: MoleculeBackgroundProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const animationRef = useRef<number>();
  const mouseRef = useRef({ x: -1000, y: -1000, isPressed: false });
  const mouseTrailRef = useRef<{ x: number; y: number; age: number }[]>([]);
  const rippleRef = useRef<{ x: number; y: number; radius: number; opacity: number }[]>([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resizeCanvas = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
    };

    const initParticles = () => {
      const rect = canvas.getBoundingClientRect();
      particlesRef.current = Array.from({ length: particleCount }, () => ({
        x: Math.random() * rect.width,
        y: Math.random() * rect.height,
        vx: (Math.random() - 0.5) * speed,
        vy: (Math.random() - 0.5) * speed,
        radius: Math.random() * 3 + 2,
        color: COLORS[Math.floor(Math.random() * COLORS.length)],
        opacity: Math.random() * 0.5 + 0.3,
        connections: [],
      }));
    };

    // Draw mouse glow cursor
    const drawMouseGlow = () => {
      if (mouseRef.current.x < 0) return;
      
      // Draw trail
      const trail = mouseTrailRef.current;
      for (let i = 0; i < trail.length; i++) {
        const point = trail[i];
        const alpha = (1 - point.age / 20) * 0.3;
        const size = (1 - point.age / 20) * 40;
        
        const gradient = ctx.createRadialGradient(
          point.x, point.y, 0,
          point.x, point.y, size
        );
        gradient.addColorStop(0, `rgba(139, 92, 246, ${alpha})`);
        gradient.addColorStop(0.5, `rgba(59, 130, 246, ${alpha * 0.5})`);
        gradient.addColorStop(1, 'transparent');
        
        ctx.beginPath();
        ctx.arc(point.x, point.y, size, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();
      }
      
      // Draw main cursor glow
      const gradient = ctx.createRadialGradient(
        mouseRef.current.x, mouseRef.current.y, 0,
        mouseRef.current.x, mouseRef.current.y, 100
      );
      gradient.addColorStop(0, mouseRef.current.isPressed ? 'rgba(236, 72, 153, 0.4)' : 'rgba(139, 92, 246, 0.3)');
      gradient.addColorStop(0.3, mouseRef.current.isPressed ? 'rgba(236, 72, 153, 0.15)' : 'rgba(59, 130, 246, 0.1)');
      gradient.addColorStop(1, 'transparent');
      
      ctx.beginPath();
      ctx.arc(mouseRef.current.x, mouseRef.current.y, 100, 0, Math.PI * 2);
      ctx.fillStyle = gradient;
      ctx.fill();
      
      // Draw ripples
      const ripples = rippleRef.current;
      for (let i = ripples.length - 1; i >= 0; i--) {
        const ripple = ripples[i];
        ctx.beginPath();
        ctx.arc(ripple.x, ripple.y, ripple.radius, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(139, 92, 246, ${ripple.opacity})`;
        ctx.lineWidth = 2;
        ctx.stroke();
        
        ripple.radius += 3;
        ripple.opacity -= 0.015;
        
        if (ripple.opacity <= 0) {
          ripples.splice(i, 1);
        }
      }
    };

    const drawMolecule = (particle: Particle, mouseDistance: number) => {
      // Enhanced glow when near mouse
      const nearMouse = mouseDistance < 200;
      const glowMultiplier = nearMouse ? 1.5 + (1 - mouseDistance / 200) : 1;
      const pulseEffect = nearMouse ? Math.sin(Date.now() / 200) * 0.2 + 1 : 1;
      
      // Glow effect
      const gradient = ctx.createRadialGradient(
        particle.x, particle.y, 0,
        particle.x, particle.y, particle.radius * 3 * glowMultiplier
      );
      gradient.addColorStop(0, particle.color);
      gradient.addColorStop(1, 'transparent');
      
      ctx.beginPath();
      ctx.arc(particle.x, particle.y, particle.radius * 3 * glowMultiplier * pulseEffect, 0, Math.PI * 2);
      ctx.fillStyle = gradient;
      ctx.globalAlpha = particle.opacity * 0.3 * glowMultiplier;
      ctx.fill();

      // Core - bigger when near mouse
      ctx.beginPath();
      ctx.arc(particle.x, particle.y, particle.radius * pulseEffect, 0, Math.PI * 2);
      ctx.fillStyle = particle.color;
      ctx.globalAlpha = Math.min(1, particle.opacity * glowMultiplier);
      ctx.fill();

      // Highlight
      ctx.beginPath();
      ctx.arc(
        particle.x - particle.radius * 0.3,
        particle.y - particle.radius * 0.3,
        particle.radius * 0.4,
        0,
        Math.PI * 2
      );
      ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
      ctx.globalAlpha = particle.opacity * 0.5;
      ctx.fill();
      
      ctx.globalAlpha = 1;
    };

    const drawConnection = (p1: Particle, p2: Particle, distance: number) => {
      const opacity = 1 - distance / connectionDistance;
      
      // Create gradient for bond
      const gradient = ctx.createLinearGradient(p1.x, p1.y, p2.x, p2.y);
      gradient.addColorStop(0, p1.color.replace('0.8', `${opacity * 0.4}`).replace('0.7', `${opacity * 0.4}`).replace('0.6', `${opacity * 0.4}`));
      gradient.addColorStop(1, p2.color.replace('0.8', `${opacity * 0.4}`).replace('0.7', `${opacity * 0.4}`).replace('0.6', `${opacity * 0.4}`));

      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.strokeStyle = gradient;
      ctx.lineWidth = opacity * 2;
      ctx.stroke();

      // Draw bond midpoint (electron pair effect)
      if (opacity > 0.5) {
        const midX = (p1.x + p2.x) / 2;
        const midY = (p1.y + p2.y) / 2;
        ctx.beginPath();
        ctx.arc(midX, midY, opacity * 2, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${opacity * 0.3})`;
        ctx.fill();
      }
    };

    const animate = () => {
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);

      const particles = particlesRef.current;
      
      // Update mouse trail
      if (mouseRef.current.x > 0) {
        mouseTrailRef.current.unshift({ x: mouseRef.current.x, y: mouseRef.current.y, age: 0 });
        if (mouseTrailRef.current.length > 20) {
          mouseTrailRef.current.pop();
        }
        mouseTrailRef.current.forEach(p => p.age++);
      }
      
      // Draw mouse glow first (behind particles)
      drawMouseGlow();

      // Update positions
      particles.forEach((particle) => {
        // Mouse interaction - STRONG repulsion/attraction effect
        const dx = mouseRef.current.x - particle.x;
        const dy = mouseRef.current.y - particle.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        
        const interactionRadius = 250;
        if (dist < interactionRadius && dist > 0) {
          const force = Math.pow((interactionRadius - dist) / interactionRadius, 2);
          
          if (mouseRef.current.isPressed) {
            // Attract when clicked
            particle.vx += (dx / dist) * force * 0.15;
            particle.vy += (dy / dist) * force * 0.15;
          } else {
            // Repel normally with swirl effect
            const angle = Math.atan2(dy, dx);
            const swirlAngle = angle + Math.PI / 2;
            particle.vx -= (dx / dist) * force * 0.08;
            particle.vy -= (dy / dist) * force * 0.08;
            // Add slight swirl
            particle.vx += Math.cos(swirlAngle) * force * 0.03;
            particle.vy += Math.sin(swirlAngle) * force * 0.03;
          }
        }

        particle.x += particle.vx;
        particle.y += particle.vy;

        // Bounce off edges
        if (particle.x < 0 || particle.x > rect.width) {
          particle.vx *= -1;
          particle.x = Math.max(0, Math.min(rect.width, particle.x));
        }
        if (particle.y < 0 || particle.y > rect.height) {
          particle.vy *= -1;
          particle.y = Math.max(0, Math.min(rect.height, particle.y));
        }

        // Add some random movement
        particle.vx += (Math.random() - 0.5) * 0.01;
        particle.vy += (Math.random() - 0.5) * 0.01;

        // Apply friction for smoother movement
        particle.vx *= 0.99;
        particle.vy *= 0.99;

        // Limit velocity
        const maxSpeed = speed * 3;
        const currentSpeed = Math.sqrt(particle.vx * particle.vx + particle.vy * particle.vy);
        if (currentSpeed > maxSpeed) {
          particle.vx = (particle.vx / currentSpeed) * maxSpeed;
          particle.vy = (particle.vy / currentSpeed) * maxSpeed;
        }
      });

      // Draw connections first (behind particles)
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < connectionDistance) {
            drawConnection(particles[i], particles[j], distance);
          }
        }
      }

      // Draw particles with mouse distance info
      particles.forEach((particle) => {
        const dx = mouseRef.current.x - particle.x;
        const dy = mouseRef.current.y - particle.y;
        const mouseDistance = Math.sqrt(dx * dx + dy * dy);
        drawMolecule(particle, mouseDistance);
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    const handleMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouseRef.current = {
        ...mouseRef.current,
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      };
    };

    const handleMouseDown = () => {
      mouseRef.current.isPressed = true;
      // Add ripple effect on click
      if (mouseRef.current.x > 0) {
        rippleRef.current.push({
          x: mouseRef.current.x,
          y: mouseRef.current.y,
          radius: 10,
          opacity: 0.6,
        });
      }
    };

    const handleMouseUp = () => {
      mouseRef.current.isPressed = false;
    };

    const handleMouseLeave = () => {
      mouseRef.current = { x: -1000, y: -1000, isPressed: false };
      mouseTrailRef.current = [];
    };

    resizeCanvas();
    initParticles();
    animate();

    window.addEventListener('resize', () => {
      resizeCanvas();
      initParticles();
    });
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mouseup', handleMouseUp);
    canvas.addEventListener('mouseleave', handleMouseLeave);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      window.removeEventListener('resize', resizeCanvas);
      canvas.removeEventListener('mousemove', handleMouseMove);
      canvas.removeEventListener('mousedown', handleMouseDown);
      canvas.removeEventListener('mouseup', handleMouseUp);
      canvas.removeEventListener('mouseleave', handleMouseLeave);
    };
  }, [particleCount, connectionDistance, speed]);

  return (
    <canvas
      ref={canvasRef}
      className={`absolute inset-0 pointer-events-auto ${className}`}
      style={{ width: '100%', height: '100%' }}
    />
  );
}

// Floating orbs for a more subtle effect
export function FloatingOrbs({ className = '' }: { className?: string }) {
  return (
    <div className={`absolute inset-0 overflow-hidden pointer-events-none ${className}`}>
      {/* Large gradient orbs */}
      <div className="absolute top-1/4 -left-32 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl animate-float" style={{ animationDelay: '0s', animationDuration: '8s' }} />
      <div className="absolute top-1/2 -right-32 w-80 h-80 bg-purple-500/20 rounded-full blur-3xl animate-float" style={{ animationDelay: '2s', animationDuration: '10s' }} />
      <div className="absolute -bottom-20 left-1/3 w-72 h-72 bg-cyan-500/15 rounded-full blur-3xl animate-float" style={{ animationDelay: '4s', animationDuration: '12s' }} />
      <div className="absolute top-10 right-1/4 w-64 h-64 bg-pink-500/10 rounded-full blur-3xl animate-float" style={{ animationDelay: '1s', animationDuration: '9s' }} />
      
      {/* Smaller accent orbs */}
      <div className="absolute top-1/3 left-1/4 w-32 h-32 bg-emerald-500/20 rounded-full blur-2xl animate-pulse-glow" style={{ animationDelay: '0.5s' }} />
      <div className="absolute bottom-1/4 right-1/3 w-40 h-40 bg-amber-500/15 rounded-full blur-2xl animate-pulse-glow" style={{ animationDelay: '1.5s' }} />
    </div>
  );
}

// Grid pattern with glow
export function GridPattern({ className = '' }: { className?: string }) {
  return (
    <div className={`absolute inset-0 pointer-events-none ${className}`}>
      {/* Base grid */}
      <div 
        className="absolute inset-0 opacity-30"
        style={{
          backgroundImage: `
            linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)
          `,
          backgroundSize: '64px 64px',
        }}
      />
      {/* Radial fade */}
      <div 
        className="absolute inset-0"
        style={{
          background: 'radial-gradient(ellipse at 50% 50%, transparent 0%, rgba(10, 10, 15, 0.8) 70%)',
        }}
      />
    </div>
  );
}

// Combined premium background
export function PremiumBackground({ 
  showMolecules = true,
  showOrbs = true,
  showGrid = true,
  particleCount = 40,
}: { 
  showMolecules?: boolean;
  showOrbs?: boolean;
  showGrid?: boolean;
  particleCount?: number;
}) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div className="fixed inset-0 z-0 overflow-hidden bg-[#0a0a0f]">
      {/* Base gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-950/30 via-transparent to-purple-950/30" />
      
      {/* Grid pattern */}
      {showGrid && <GridPattern />}
      
      {/* Floating orbs */}
      {showOrbs && <FloatingOrbs />}
      
      {/* Molecule animation */}
      {showMolecules && mounted && (
        <MoleculeBackground 
          particleCount={particleCount} 
          connectionDistance={120}
          speed={0.3}
        />
      )}
      
      {/* Noise texture overlay */}
      <div 
        className="absolute inset-0 opacity-[0.015] pointer-events-none"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
        }}
      />
      
      {/* Vignette */}
      <div 
        className="absolute inset-0 pointer-events-none"
        style={{
          background: 'radial-gradient(ellipse at 50% 50%, transparent 0%, rgba(0, 0, 0, 0.4) 100%)',
        }}
      />
    </div>
  );
}
