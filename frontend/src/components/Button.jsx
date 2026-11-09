import React from 'react';

export function Button({
  children,
  variant = 'primary',
  size = 'md',
  disabled = false,
  loading = false,
  onClick,
  type = 'button',
  className = '',
  ...props
}) {
  return (
    <button
      type={type}
      className={`btn btn-${variant} btn-${size} ${className} ${loading ? 'loading' : ''} ${disabled ? 'disabled' : ''}`}
      onClick={onClick}
      disabled={disabled || loading}
      {...props}
    >
      {loading && <span className="btn-spinner" />}
      {children}
    </button>
  );
}

export function ButtonGroup({ children, className = '' }) {
  return <div className={`btn-group ${className}`}>{children}</div>;
}