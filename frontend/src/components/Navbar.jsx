import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Zap, LayoutDashboard, TrendingUp } from 'lucide-react';

function getIST() {
  return new Date().toLocaleString('en-IN', {
    timeZone: 'Asia/Kolkata',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
    hour12: true,
  });
}
function getISTDate() {
  return new Date().toLocaleDateString('en-IN', {
    timeZone: 'Asia/Kolkata',
    weekday: 'short', day: 'numeric', month: 'short', year: 'numeric',
  });
}

export default function Navbar() {
  const [time, setTime] = useState(getIST());
  const [date, setDate] = useState(getISTDate());
  const location = useLocation();

  useEffect(() => {
    const id = setInterval(() => {
      setTime(getIST());
      setDate(getISTDate());
    }, 1000);
    return () => clearInterval(id);
  }, []);

  const navLink = (to, label, Icon) => {
    const active = location.pathname === to;
    return (
      <Link
        to={to}
        className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all duration-200 ${
          active
            ? 'bg-white/20 text-white shadow-md'
            : 'text-indigo-100 hover:bg-white/10 hover:text-white'
        }`}
      >
        <Icon size={16} />
        {label}
      </Link>
    );
  };

  return (
    <nav className="sticky top-0 z-50 bg-gradient-to-r from-indigo-700 via-indigo-600 to-sky-600 shadow-lg">
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="relative w-10 h-10 flex items-center justify-center">
            <div className="absolute inset-0 bg-white/20 rounded-xl" />
            <Zap size={22} className="text-yellow-300 relative z-10" strokeWidth={2.5} />
          </div>
          <div>
            <div className="text-xl font-black tracking-tight leading-none">
              <span className="text-indigo-100">E</span>
              <span className="text-white">-</span>
              <span className="text-sky-200">City</span>
            </div>
            <div className="text-indigo-200 text-xs font-medium">Powering India's Grid Intelligence</div>
          </div>
        </div>

        {/* Navigation */}
        <div className="flex items-center gap-2">
          {navLink('/', 'Dashboard', LayoutDashboard)}
          {navLink('/forecast', 'Forecast', TrendingUp)}
        </div>

        {/* IST Clock */}
        <div className="text-right">
          <div className="text-white font-mono font-bold text-lg leading-none">{time}</div>
          <div className="text-indigo-200 text-xs mt-0.5">{date} IST</div>
        </div>
      </div>
    </nav>
  );
}

