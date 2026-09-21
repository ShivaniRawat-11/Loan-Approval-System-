const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

function getAuthHeader() {
  const stored = localStorage.getItem('user');
  if (stored) {
    const user = JSON.parse(stored);
    if (user.token) {
      return { 'Authorization': `Bearer ${user.token}` };
    }
  }
  return {};
}

export async function login(email, password) {
  const res = await fetch(`${API_URL}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  return res.json();
}

export async function signup(email, password, name) {
  const res = await fetch(`${API_URL}/api/auth/signup`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password, name }),
  });
  return res.json();
}

export async function sendMessage(message, mode, chatHistory) {
  const res = await fetch(`${API_URL}/api/chat/send`, {
    method: 'POST',
    headers: { 
      'Content-Type': 'application/json',
      ...getAuthHeader()
    },
    body: JSON.stringify({
      message,
      mode,
      chat_history: chatHistory,
    }),
  });
  if (res.status === 401) throw new Error('Unauthorized');
  return res.json();
}

export async function streamMessage(message, mode, chatHistory) {
  const res = await fetch(`${API_URL}/api/chat/stream`, {
    method: 'POST',
    headers: { 
      'Content-Type': 'application/json',
      ...getAuthHeader()
    },
    body: JSON.stringify({
      message,
      mode,
      chat_history: chatHistory,
    }),
  });
  if (res.status === 401) throw new Error('Unauthorized');
  return res;
}

export async function getChatHistory() {
  const res = await fetch(`${API_URL}/api/chat/history`, {
    headers: { ...getAuthHeader() }
  });
  if (res.status === 401) throw new Error('Unauthorized');
  return res.json();
}

export async function clearChatHistory() {
  const res = await fetch(`${API_URL}/api/chat/clear`, {
    method: 'POST',
    headers: { ...getAuthHeader() }
  });
  if (res.status === 401) throw new Error('Unauthorized');
  return res.json();
}

export async function processDocuments(files) {
  const formData = new FormData();
  for (const file of files) {
    formData.append('files', file);
  }
  const res = await fetch(`${API_URL}/api/documents/process`, {
    method: 'POST',
    headers: { ...getAuthHeader() },
    body: formData,
  });
  if (res.status === 401) throw new Error('Unauthorized');
  return res.json();
}

export async function healthCheck() {
  try {
    const res = await fetch(`${API_URL}/api/health`);
    return res.json();
  } catch {
    return { status: 'error' };
  }
}
