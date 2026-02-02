/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  images: {
    domains: ["localhost"],
  },
  // Enable standalone output for Docker
  output: "standalone",
};

module.exports = nextConfig;
