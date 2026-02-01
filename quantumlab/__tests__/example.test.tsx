import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { SessionProvider } from 'next-auth/react';
import '@testing-library/jest-dom';

// Mock next-auth
jest.mock('next-auth/react', () => ({
  SessionProvider: ({ children }: { children: React.ReactNode }) => children,
  useSession: jest.fn(() => ({
    data: { user: { email: 'test@example.com' } },
    status: 'authenticated',
  })),
}));

// Mock next/navigation
jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: jest.fn(),
    replace: jest.fn(),
    prefetch: jest.fn(),
  }),
  usePathname: () => '/',
}));

describe('Example Test Suite', () => {
  it('should render without crashing', () => {
    const { container } = render(<div>Hello World</div>);
    expect(container).toBeInTheDocument();
  });

  it('should find text content', () => {
    render(<div>Hello World</div>);
    expect(screen.getByText('Hello World')).toBeInTheDocument();
  });
});

// Add more tests for your components here
