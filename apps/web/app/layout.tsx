import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "My Strava | Ride intelligence",
  description: "A personal dashboard for training, rides, and equipment.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
