import React from 'react';

export function Card({ children, className = '', style, ...props }) {
  return (
    <div
      className={`card ${className}`}
      style={style}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardHeader({ title, subtitle, children, className = '' }) {
  return (
    <div className={`card-header ${className}`}>
      <div>
        {title && <h3 className="card-title">{title}</h3>}
        {subtitle && <p className="card-subtitle">{subtitle}</p>}
      </div>
      {children}
    </div>
  );
}

export function CardContent({ children, className = '' }) {
  return <div className={`card-content ${className}`}>{children}</div>;
}

export function CardFooter({ children, className = '' }) {
  return <div className={`card-footer ${className}`}>{children}</div>;
}