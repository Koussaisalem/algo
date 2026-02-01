# QuantumLab 

> **End-to-end platform for quantum materials discovery combining generative AI, quantum chemistry, and intuitive visualization**

[![Next.js](https://img.shields.io/badge/Next.js-14-black?logo=next.js)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-blue?logo=typescript)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3-38bdf8?logo=tailwind-css)](https://tailwindcss.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](../LICENSE)

---

## Overview

**QuantumLab** is a professional-grade web platform that makes quantum materials discovery accessible to researchers worldwide. Built with Apple-style design principles and powered by the proven QCMD-ECS framework, it provides an intuitive interface for the complete discovery pipeline.

### Key Features

- **Apple-Style Glassmorphism UI** - Beautiful, modern interface with liquid glass effects
- **Custom Dataset Management** - Upload, organize, and visualize molecular datasets
- **AI Model Training** - Configure and train custom generative models
- **DFT Computations** - Run xTB or full DFT validations
- **Real-time Monitoring** - Track computation progress with live updates
- **3D Visualization** - Interactive molecular structure viewer
- **Export & Share** - Download results and share discoveries

---

## Quick Start

### Prerequisites

- Node.js 18+ and npm/yarn/pnpm
- Python 3.10+ (for backend integration)
- PostgreSQL (for production)

### Installation

```bash
# Navigate to the platform directory
cd quantumlab

# Install dependencies
npm install
# or
yarn install
# or
pnpm install

# Run development server
npm run dev

# Open http://localhost:3000
```

### Environment Variables

Create a `.env.local` file:

```env
# Database
DATABASE_URL="postgresql://user:password@localhost:5432/quantumlab"

# Python Backend
PYTHON_API_URL="http://localhost:8000"

# Optional: Authentication
NEXTAUTH_SECRET="your-secret-key"
NEXTAUTH_URL="http://localhost:3000"
```

---

## Project Structure

```
quantumlab/
 app/ # Next.js App Router
 (auth)/ # Authentication pages
 dashboard/ # Main dashboard
 datasets/ # Dataset management
 models/ # Model training
 compute/ # DFT computations
 results/ # Results & visualizations
 layout.tsx # Root layout
 page.tsx # Landing page
 globals.css # Global styles
 components/
 ui/ # shadcn/ui components
 dashboard/ # Dashboard components
 datasets/ # Dataset components
 models/ # Model components
 compute/ # Computation components
 visualizations/ # 3D viewers, charts
 lib/
 api/ # API client functions
 utils.ts # Utility functions
 types.ts # TypeScript types
 hooks/ # Custom React hooks
 public/ # Static assets
 styles/ # Additional styles
```

---

## Design System

### Color Palette

```css
/* Primary */
--blue-500: #3b82f6
--purple-500: #a855f7
--pink-500: #ec4899

/* Glassmorphism */
background: rgba(255, 255, 255, 0.7)
backdrop-filter: blur(20px)
border: 1px solid rgba(255, 255, 255, 0.2)
```

### Components

Built with shadcn/ui and custom Apple-style variants:

- **Glass Cards** - Translucent cards with blur effects
- **Gradient Buttons** - Smooth gradient transitions
- **Animated Inputs** - Fluid input interactions
- **Toast Notifications** - Non-intrusive notifications
- **Modal Dialogs** - Smooth modal transitions

---

## Core Features

### 1. Dataset Management

**Upload & Organize**
- Drag-and-drop file upload
- Support for multiple formats (.pt, .xyz, .sdf)
- Automatic data validation
- Visual dataset preview

**Features:**
```typescript
// Upload dataset
const uploadDataset = async (file: File) => {
 const formData = new FormData()
 formData.append('file', file)
 
 const response = await fetch('/api/datasets/upload', {
 method: 'POST',
 body: formData
 })
 
 return response.json()
}
```

### 2. Model Training

**Configure & Train**
- Interactive model architecture builder
- Hyperparameter tuning interface
- Training progress visualization
- Model comparison tools

**Supported Models:**
- Surrogate models (GNN-based)
- Score models (diffusion-based)
- Custom architectures

### 3. DFT Computations

**Run Calculations**
- xTB (fast, semi-empirical)
- GPAW (accurate, DFT)
- Custom computational settings
- Batch processing support

### 4. Results & Visualization

**Interactive Analysis**
- 3D molecular viewer
- Property plots and charts
- Comparison tools
- Export to various formats

---

## API Integration

### Python Backend Communication

```typescript
// lib/api/python-client.ts
import axios from 'axios'

const pythonAPI = axios.create({
 baseURL: process.env.PYTHON_API_URL,
 timeout: 30000,
})

export const runDFT = async (moleculeId: string, method: 'xtb' | 'dft') => {
 const response = await pythonAPI.post('/compute/dft', {
 molecule_id: moleculeId,
 method: method,
 })
 return response.data
}
```

### API Routes

- `POST /api/datasets/upload` - Upload dataset
- `GET /api/datasets` - List datasets
- `POST /api/models/train` - Start training
- `GET /api/models/:id/status` - Training status
- `POST /api/compute/dft` - Run DFT calculation
- `GET /api/results/:id` - Get results

---

## Workflows

### Complete Discovery Pipeline

1. **Upload Dataset**
 ```typescript
 const dataset = await uploadDataset(file)
 ```

2. **Train Model**
 ```typescript
 const model = await trainModel({
 datasetId: dataset.id,
 modelType: 'score',
 epochs: 100
 })
 ```

3. **Generate Molecules**
 ```typescript
 const generated = await generateMolecules({
 modelId: model.id,
 numSamples: 100
 })
 ```

4. **Validate with DFT**
 ```typescript
 const results = await runDFT(generated[0].id, 'xtb')
 ```

5. **Visualize & Export**
 ```typescript
 visualizeMolecule(results.structure)
 exportResults(results, 'json')
 ```

---

## Development

### Local Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Run type checking
npm run type-check

# Run linting
npm run lint

# Build for production
npm run build

# Start production server
npm run start
```

### Adding New Components

```bash
# Add shadcn/ui component
npx shadcn-ui@latest add [component-name]

# Example: Add dialog component
npx shadcn-ui@latest add dialog
```

### Custom Hooks

```typescript
// hooks/use-dataset.ts
import { useState, useEffect } from 'react'

export function useDataset(id: string) {
 const [dataset, setDataset] = useState(null)
 const [loading, setLoading] = useState(true)

 useEffect(() => {
 fetch(`/api/datasets/${id}`)
 .then(res => res.json())
 .then(data => {
 setDataset(data)
 setLoading(false)
 })
 }, [id])

 return { dataset, loading }
}
```

---

## Performance

- **Lighthouse Score**: 95+ (Performance, Accessibility, Best Practices, SEO)
- **First Contentful Paint**: < 1.5s
- **Time to Interactive**: < 3.0s
- **Bundle Size**: Optimized with tree-shaking and code splitting

---

## Security

- **Authentication**: NextAuth.js with multiple providers
- **Authorization**: Role-based access control (RBAC)
- **Data Encryption**: TLS/SSL for all communications
- **Input Validation**: Zod schemas for all inputs
- **CSRF Protection**: Built-in Next.js protection

---

## Deployment

### Vercel (Recommended)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

### Docker

```dockerfile
# Dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
```

### Environment Variables for Production

```env
DATABASE_URL=
PYTHON_API_URL=
NEXTAUTH_SECRET=
NEXTAUTH_URL=
```

---

## Contributing

We welcome contributions from the community!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Documentation

- [User Guide](./docs/USER_GUIDE.md)
- [API Reference](./docs/API.md)
- [Design System](./docs/DESIGN.md)
- [Deployment Guide](./docs/DEPLOYMENT.md)

---

## Troubleshooting

### Common Issues

**Build Errors**
```bash
# Clear cache and reinstall
rm -rf node_modules .next
npm install
npm run build
```

**API Connection Failed**
```bash
# Check Python backend is running
# Update PYTHON_API_URL in .env.local
```

**Styling Issues**
```bash
# Rebuild Tailwind
npm run dev
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

## Acknowledgments

- Built with [Next.js](https://nextjs.org/)
- UI components from [shadcn/ui](https://ui.shadcn.com/)
- Styled with [Tailwind CSS](https://tailwindcss.com/)
- Icons from [Lucide](https://lucide.dev/)
- Powered by the QCMD-ECS framework

---

## Contact

For questions, issues, or collaboration:
- **GitHub Issues**: [Report a bug](https://github.com/Koussaisalem/algo/issues)
- **Email**: your-email@example.com

---

<div align="center">

**[ back to top](#quantumlab-)**

Made with by the QuantumLab Team

</div>
