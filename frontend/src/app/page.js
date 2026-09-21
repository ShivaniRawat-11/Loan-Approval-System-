'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { login, signup } from '@/lib/api';

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    if (!email || !password) {
      setError('Please enter both email and password.');
      return;
    }
    setLoading(true);
    try {
      const data = await login(email, password);
      if (data.success) {
        localStorage.setItem('user', JSON.stringify({ 
          email: data.email, 
          name: data.name,
          token: data.token
        }));
        router.push('/dashboard');
      } else {
        setError(data.message || 'Login failed.');
      }
    } catch (err) {
      setError('Network error. Please check if the server is running.');
    }
    setLoading(false);
  };

  const handleSignup = async () => {
    setError('');
    setSuccess('');
    if (!email || !password) {
      setError('Please enter both email and password to sign up.');
      return;
    }
    if (password.length < 6) {
      setError('Password must be at least 6 characters.');
      return;
    }
    setLoading(true);
    try {
      const data = await signup(email, password);
      if (data.success) {
        setSuccess(`${data.message} You can now Sign In.`);
      } else {
        setError(data.message || 'Signup failed.');
      }
    } catch (err) {
      setError('Network error. Please check if the server is running.');
    }
    setLoading(false);
  };

  return (
    <div className="login-container">
      {/* Left Branding Panel */}
      <div className="login-left-panel">
        {/* Decorative elements */}
        <div style={{position:'absolute',width:'200px',height:'200px',border:'1px solid rgba(255,255,255,0.04)',borderRadius:'50%',bottom:'15%',left:'-40px'}} />
        
        <div className="left-branding">
          <div className="brand-icon-wrap">
            <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
              <rect x="9" y="10" width="6" height="5" rx="1"/>
              <path d="M10 10V8a2 2 0 0 1 4 0v2"/>
            </svg>
          </div>
          <h1 className="brand-title">Secure Loan<br/><span>Portal</span></h1>
          <p className="brand-subtitle">
            Enterprise-grade AI loan approval system with real-time document analysis and intelligent decision making.
          </p>
          <ul className="feature-list">
            <li className="feature-item">
              <div className="feature-icon fi-blue">🤖</div>
              <span className="feature-text">AI-Powered Loan Assessment</span>
            </li>
            <li className="feature-item">
              <div className="feature-icon fi-indigo">📄</div>
              <span className="feature-text">Instant Document Analysis</span>
            </li>
            <li className="feature-item">
              <div className="feature-icon fi-cyan">⚡</div>
              <span className="feature-text">Real-time Eligibility Scoring</span>
            </li>
          </ul>
          <div className="trust-badge">
            <div className="trust-dot" />
            <span>SOC 2 Compliant &nbsp;·&nbsp; 256-bit Encryption &nbsp;·&nbsp; Enterprise Ready</span>
          </div>
        </div>
      </div>

      {/* Right Form Panel */}
      <div className="login-right-panel">
        <div>
          <form className="form-card" onSubmit={handleLogin}>
            <div className="form-header">
              <h2>Welcome back</h2>
              <p>Sign in to your enterprise account</p>
            </div>

            {error && <div className="alert-error">{error}</div>}
            {success && <div className="alert-success">{success}</div>}

            <label className="form-label">Email Address</label>
            <input
              id="login-email"
              type="text"
              className="form-input"
              placeholder="name@company.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="email"
            />

            <label className="form-label">Password</label>
            <input
              id="login-password"
              type="password"
              className="form-input"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
            />

            <div className="btn-row">
              <button id="sign-in-btn" type="submit" className="btn-primary" disabled={loading}>
                {loading ? <><span className="spinner" /> Signing in...</> : 'Sign In'}
              </button>
              <button
                id="create-account-btn"
                type="button"
                className="btn-secondary"
                onClick={handleSignup}
                disabled={loading}
              >
                Create Account
              </button>
            </div>
          </form>

          <div className="security-footer">
            <p>
              <span className="security-dot" />
              Protected by 256-bit SSL encryption
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
