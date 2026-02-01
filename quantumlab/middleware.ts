export { default } from 'next-auth/middleware';

// Protect these routes - require authentication
export const config = {
  matcher: [
    '/dashboard/:path*',
    '/datasets/:path*',
    '/models/:path*',
    '/inference/:path*',
    '/library/:path*',
    '/cloud/:path*',
    '/compute/:path*',
    '/results/:path*',
  ],
};
