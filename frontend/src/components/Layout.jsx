import React from 'react';
import { Outlet, NavLink, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  Upload,
  Search,
  HelpCircle,
  FileText,
  Database,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import './Layout.css';

const navItems = [
  { path: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { path: 'ingest', label: 'Ingest', icon: Upload },
  { path: 'search', label: 'Search', icon: Search },
  { path: 'ask', label: 'Ask', icon: HelpCircle },
  { path: 'documents', label: 'Documents', icon: FileText },
  { path: 'extraction', label: 'Extraction', icon: Database },
];

export default function Layout() {
  const [collapsed, setCollapsed] = React.useState(false);

  return (
    <div className="layout">
      <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
        <div className="sidebar-header">
          <h1 className={collapsed ? 'hidden' : ''}>RFP Intelligence</h1>
          <button
            className="collapse-btn"
            onClick={() => setCollapsed(!collapsed)}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
          </button>
        </div>
        <nav className="sidebar-nav">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
                title={collapsed ? item.label : ''}
              >
                <Icon size={20} />
                {!collapsed && <span>{item.label}</span>}
              </NavLink>
            );
          })}
        </nav>
        <div className="sidebar-footer">
          <span className="version">v1.0.0</span>
        </div>
      </aside>
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  );
}