import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Ingest from './pages/Ingest';
import Search from './pages/Search';
import Ask from './pages/Ask';
import Documents from './pages/Documents';
import Extraction from './pages/Extraction';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="ingest" element={<Ingest />} />
          <Route path="search" element={<Search />} />
          <Route path="ask" element={<Ask />} />
          <Route path="documents" element={<Documents />} />
          <Route path="extraction" element={<Extraction />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;