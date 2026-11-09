import axios from 'axios';

// Use relative URL in development (proxied by Vite), absolute in production
const API_BASE = import.meta.env.DEV ? '' : 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Health check
export const healthCheck = () => api.get('/health');

// Bids
export const listBids = () => api.get('/bids');

// Ingest
export const ingestDocuments = (bidId, folderPath) =>
  api.post('/ingest', { bid_id: bidId, folder_path: folderPath });

// Index
export const indexDocuments = (bidId, force = false) =>
  api.post('/index', { bid_id: bidId, force });

// Search
export const searchDocuments = (query, bidIds, topK = 5, mode = 'hybrid_rerank', signal) =>
  api.post('/search', { query, bid_ids: bidIds || [], top_k: topK, mode }, { signal });

// Ask
export const askQuestion = (bidId, question, topK = 5, mode = 'fusion') =>
  api.post('/ask', { bid_id: bidId, question, top_k: topK, mode });

// Extract
export const extractFields = (bidId) => api.post(`/extract/${bidId}`);

// Documents
export const listDocuments = (bidId) => api.get(`/documents/${bidId}`);
export const getDocumentChunks = (bidId, fileName) => api.get(`/chunks/${bidId}/${fileName}`);

// Upload
export const uploadFiles = (bidId, files) => {
  const formData = new FormData();
  formData.append('bid_id', bidId);
  files.forEach(file => formData.append('files', file));
  return api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export default api;