import type { Metadata } from "next"
import "./globals.css"

export const metadata: Metadata = {
  title: "SiloQ - Personal AI Data Assistant",
  description: "Chat with your data from local or cloud storage",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="bg-secondary text-primary">{children}</body>
    </html>
  )
}
