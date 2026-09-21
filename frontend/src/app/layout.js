import './globals.css';

export const metadata = {
  title: 'Secure Loan Portal',
  description: 'Enterprise-grade AI loan approval system with real-time document analysis and intelligent decision making.',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
