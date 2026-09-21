'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { streamMessage, getChatHistory, clearChatHistory, processDocuments } from '@/lib/api';

export default function DashboardPage() {
  const router = useRouter();
  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // User state
  const [user, setUser] = useState(null);
  
  // Chat state
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [mode, setMode] = useState('general');

  // Document state
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [docsProcessed, setDocsProcessed] = useState(false);
  const [processedFileNames, setProcessedFileNames] = useState([]);
  const [documentAnalyses, setDocumentAnalyses] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);

  // Auth check
  useEffect(() => {
    const stored = localStorage.getItem('user');
    if (!stored) {
      router.push('/');
      return;
    }
    const userData = JSON.parse(stored);
    setUser(userData);
    // Load chat history
    getChatHistory().then((data) => {
      if (data.messages) {
        setMessages(data.messages);
      }
    }).catch((err) => {
      if (err.message === 'Unauthorized') handleLogout();
    });
  }, [router]);

  // Auto-scroll
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleLogout = () => {
    localStorage.removeItem('user');
    router.push('/');
  };

  const handleClearChat = async () => {
    if (user) {
      try {
        await clearChatHistory();
        setMessages([]);
      } catch (err) {
        if (err.message === 'Unauthorized') handleLogout();
      }
    }
  };

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    setSelectedFiles(files);
  };

  const handleProcessDocuments = async () => {
    if (!selectedFiles.length || !user) return;
    setIsProcessing(true);
    try {
      const data = await processDocuments(selectedFiles);
      if (data.success) {
        setDocsProcessed(true);
        setProcessedFileNames(data.file_names);
        setDocumentAnalyses(data.document_analyses || []);
        setMessages([]);
      } else {
        alert(data.detail || 'Document processing failed.');
      }
    } catch (err) {
      if (err.message === 'Unauthorized') {
        handleLogout();
      } else {
        alert('Document processing failed. Check server logs.');
      }
    }
    setIsProcessing(false);
  };

  const handleSendMessage = async (messageText) => {
    const text = messageText || input.trim();
    if (!text || isStreaming || !user) return;

    const userMsg = { role: 'user', content: text };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput('');
    setIsStreaming(true);

    // Add placeholder for assistant
    const assistantMsg = { role: 'assistant', content: '' };
    setMessages([...newMessages, assistantMsg]);

    try {
      const chatHistory = messages.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const response = await streamMessage(text, mode, chatHistory);
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulated = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6).trim();
            if (data === '[DONE]') continue;
            try {
              const parsed = JSON.parse(data);
              if (parsed.text) {
                accumulated += parsed.text;
                setMessages((prev) => {
                  const updated = [...prev];
                  updated[updated.length - 1] = {
                    role: 'assistant',
                    content: accumulated,
                  };
                  return updated;
                });
              }
            } catch {
              // skip invalid JSON
            }
          }
        }
      }
    } catch (err) {
      if (err.message === 'Unauthorized') {
        handleLogout();
        return;
      }
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: 'assistant',
          content: '⚠️ Error connecting to the server. Please try again.',
        };
        return updated;
      });
    }

    setIsStreaming(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const quickQueries = [
    'What is the minimum credit score required?',
    'What are the interest rates?',
    'How long is the loan approval process?',
  ];

  if (!user) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', background: '#f8fafc' }}>
        <div className="spinner" style={{ borderTopColor: '#2563eb', width: 40, height: 40 }} />
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      {/* ═══════ SIDEBAR ═══════ */}
      <aside className="sidebar">
        <div className="sidebar-welcome">
          <h3>Welcome, {user.name}</h3>
        </div>

        <div className="sidebar-brand">
          <span className="sidebar-brand-icon">🏠</span>
          <span className="sidebar-brand-text">AI Loan Approval<br/>System</span>
        </div>

        <hr className="sidebar-divider" />

        {/* File Upload */}
        <div className="sidebar-section-title">📤 Upload Documents</div>
        <div
          className="file-upload-area"
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            className="file-upload-input"
            accept=".pdf,.txt"
            multiple
            onChange={handleFileSelect}
          />
          <div className="file-upload-label">
            📁 Click to upload PDF or TXT files
          </div>
        </div>

        {selectedFiles.length > 0 && (
          <div style={{ marginBottom: 12 }}>
            {selectedFiles.map((f, i) => (
              <div key={i} className="file-list-item">
                {f.name.endsWith('.pdf') ? '📕' : '📄'} {f.name}
              </div>
            ))}
          </div>
        )}

        <button
          className="sidebar-btn"
          onClick={handleProcessDocuments}
          disabled={!selectedFiles.length || isProcessing}
          style={{ background: selectedFiles.length ? '#2563eb' : undefined, color: selectedFiles.length ? 'white' : undefined, borderColor: selectedFiles.length ? '#2563eb' : undefined }}
        >
          {isProcessing ? <><span className="spinner" /> Processing...</> : '⚡ Process Documents'}
        </button>

        {docsProcessed && (
          <>
            <div className="sidebar-section-title" style={{ marginTop: 16 }}>✅ Active Documents</div>
            {processedFileNames.map((fname, i) => (
              <div key={i} className="file-list-item">📕 {fname}</div>
            ))}
          </>
        )}

        <hr className="sidebar-divider" />

        {/* Quick Queries */}
        <div className="sidebar-section-title">💡 Quick Queries</div>
        {quickQueries.map((q, i) => (
          <button
            key={i}
            className="sidebar-btn"
            onClick={() => handleSendMessage(q)}
            disabled={isStreaming}
          >
            {q}
          </button>
        ))}

        <hr className="sidebar-divider" />

        {/* Bottom Buttons */}
        <div className="sidebar-bottom-btns">
          <button className="sidebar-btn" onClick={handleClearChat}>
            🗑️ Reset
          </button>
          <button className="sidebar-btn sidebar-btn-danger" onClick={handleLogout}>
            🚪 Logout
          </button>
        </div>
      </aside>

      {/* ═══════ MAIN CONTENT ═══════ */}
      <main className="main-content">
        {/* Header */}
        <div className="main-header">
          <h1>🏦 AI Loan Approval System</h1>
          <p>Ask loan questions instantly • Upload documents for personalized evaluation</p>
        </div>

        {/* Mode Cards */}
        <div className="mode-cards">
          <div className="mode-card general">
            <h4>💛 GENERAL GUIDANCE MODE</h4>
            <p>Get quick answers to general loan questions using banking knowledge — always available.</p>
          </div>
          <div className="mode-card document">
            <h4>📄 DOCUMENT-BASED ANALYSIS</h4>
            <p>Upload documents via the sidebar, process them, then switch to this mode for structured evaluation.</p>
          </div>
        </div>

        {/* Mode Selector */}
        <div className="mode-selector">
          <button
            className={`mode-option ${mode === 'general' ? 'active' : ''}`}
            onClick={() => setMode('general')}
          >
            General Guidance
          </button>
          <button
            className={`mode-option ${mode === 'document' ? 'active' : ''}`}
            onClick={() => setMode('document')}
          >
            Document-Based Analysis
          </button>
        </div>

        {/* Mode Status */}
        {mode === 'general' ? (
          <div className="mode-status general">
            💛 <strong>GENERAL GUIDANCE MODE — ACTIVE</strong> — Ask general loan and banking questions.
          </div>
        ) : docsProcessed ? (
          <div className="mode-status document-ready">
            📄 <strong>Document-Based Analysis — Active</strong> — {processedFileNames.length} document(s) loaded and indexed.
          </div>
        ) : (
          <div className="mode-status document-needed">
            📄 <strong>Document-Based Analysis — Documents Required</strong> — Upload and process documents first.
          </div>
        )}

        {/* Document Analysis Report */}
        {documentAnalyses.length > 0 && (
          <div className="doc-report">
            <h2>📋 Document Analysis Report</h2>
            {documentAnalyses.map((analysis, i) => {
              const doctype = analysis.type || 'UNKNOWN';
              const fname = analysis.filename || 'Document';
              const data = analysis.data || {};

              return (
                <div key={i} className="doc-card">
                  <div className="doc-card-header">
                    <span><strong>{fname}</strong> ({doctype})</span>
                  </div>
                  <div className="doc-card-body">
                    {(doctype === 'IRRELEVANT' || doctype === 'RESUME') ? (
                      <div className="doc-warning">
                        {data.warning || '⚠️ Irrelevant Document Detected'}
                      </div>
                    ) : doctype === 'LOAN_APPLICATION' ? (
                      <div className="doc-card-grid">
                        <div className="doc-card-section">
                          <h4>💼 Financial Profile</h4>
                          <p><strong>Applicant:</strong> {data.applicant_name || 'N/A'}</p>
                          <p><strong>Age:</strong> {data.age || 'N/A'}</p>
                          <p><strong>Loan Amount:</strong> ${(data.loan_amount || 0).toLocaleString()}</p>
                          <p><strong>Purpose:</strong> {data.loan_purpose || 'N/A'}</p>
                        </div>
                        <div className="doc-card-section">
                          <h4>💰 Income Verification</h4>
                          <p><strong>Employment:</strong> {data.employment_status || 'N/A'}</p>
                          <p><strong>Income:</strong> ${(data.income || 0).toLocaleString()}</p>
                          <p><strong>Credit Score:</strong> {data.credit_score || 'N/A'}</p>
                        </div>
                        <div className="doc-card-section">
                          <h4>⚠️ Risk Factors</h4>
                          {(data.risk_factors || []).length > 0 ? (
                            <ul>{data.risk_factors.map((r, j) => <li key={j}>{r}</li>)}</ul>
                          ) : <p><em>No significant risk factors.</em></p>}
                        </div>
                        <div className="doc-card-section">
                          <h4>📊 Recommendation</h4>
                          <p style={{ color: (data.loan_recommendation || '').includes('Approve') ? 'green' : 'red', fontWeight: 700 }}>
                            {data.loan_recommendation || 'N/A'}
                          </p>
                          <p>{data.eligibility_summary || 'N/A'}</p>
                        </div>
                      </div>
                    ) : doctype === 'BANK_POLICY' ? (
                      <div>
                        <p><strong>Summary:</strong> {data.policy_summary || ''}</p>
                        <div className="doc-card-grid" style={{ marginTop: 16 }}>
                          <div className="doc-card-section">
                            <h4>🎯 Key Requirements</h4>
                            <p><strong>Credit Score:</strong> {data.credit_score_requirements || ''}</p>
                            <p><strong>Income:</strong> {data.income_requirements || ''}</p>
                            <p><strong>Limits:</strong> {data.loan_limits || ''}</p>
                          </div>
                          <div className="doc-card-section">
                            <h4>⚠️ Restrictions</h4>
                            {(data.important_restrictions || []).length > 0 ? (
                              <ul>{data.important_restrictions.map((r, j) => <li key={j}>{r}</li>)}</ul>
                            ) : <p><em>None specified.</em></p>}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="doc-warning">⚠️ Unrecognized document type.</div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Chat Area */}
        <div className="chat-area">
          <div className="chat-messages">
            {messages.map((msg, i) => (
              <div key={i} className="chat-message">
                <div className={`chat-avatar ${msg.role}`}>
                  {msg.role === 'user' ? '👦' : '🤖'}
                </div>
                <div className={`chat-bubble ${msg.role}`}>
                  <div dangerouslySetInnerHTML={{ __html: formatMarkdown(msg.content) }} />
                  {msg.role === 'assistant' && msg.content && (
                    <div className="chat-disclaimer">
                      ⚠️ Disclaimer: This is an AI-generated assessment based on standard banking guidelines. Final loan approval is subject to physical document verification and official bank policies.
                    </div>
                  )}
                </div>
              </div>
            ))}
            {isStreaming && messages[messages.length - 1]?.content === '' && (
              <div className="chat-message">
                <div className="chat-avatar assistant">🤖</div>
                <div className="chat-bubble assistant">
                  <span className="spinner" style={{ borderTopColor: '#2563eb', width: 16, height: 16 }} /> Thinking...
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          <div className="chat-input-area">
            <input
              id="chat-input"
              type="text"
              className="chat-input"
              placeholder="Ask about loan eligibility, policies, or documents..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isStreaming}
            />
            <button
              id="chat-send"
              className="chat-send-btn"
              onClick={() => handleSendMessage()}
              disabled={isStreaming || !input.trim()}
            >
              {isStreaming ? 'Sending...' : 'Send'}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

/**
 * Simple markdown formatter — converts bold, headers, lists, line breaks
 */
function formatMarkdown(text) {
  if (!text) return '';
  let html = text
    // Escape HTML
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    // Headers
    .replace(/^### (.+)$/gm, '<h4>$1</h4>')
    .replace(/^## (.+)$/gm, '<h3>$1</h3>')
    .replace(/^# (.+)$/gm, '<h3>$1</h3>')
    // Bold
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // Italic
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    // Bullet lists
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    .replace(/^(\d+)\. (.+)$/gm, '<li>$2</li>')
    // Line breaks
    .replace(/\n/g, '<br/>');

  // Wrap consecutive <li> in <ul>
  html = html.replace(/((?:<li>.*?<\/li><br\/>?)+)/g, '<ul>$1</ul>');
  html = html.replace(/<ul>(.*?)<\/ul>/gs, (match, inner) => {
    return '<ul>' + inner.replace(/<br\/>/g, '') + '</ul>';
  });

  return html;
}
