import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import { ClientProviders } from "@/components/providers/client-providers"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "QuantumLab - Material Discovery Platform",
  description: "End-to-end platform for quantum materials discovery using AI and DFT",
  icons: {
    icon: "/favicon.ico",
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} antialiased`}>
        <ClientProviders>
          <div className="min-h-screen bg-[#0a0a0f] bg-grid">
            {children}
          </div>
        </ClientProviders>
      </body>
    </html>
  )
}
