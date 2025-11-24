import React, { useState } from 'react';
import { NavLink, Link } from 'react-router-dom';
import { FaGithub, FaTwitter, FaSun, FaMoon, FaBars, FaTimes } from 'react-icons/fa';
import { useTheme } from '../context/ThemeContext';

const Navbar = () => {
  const { theme, toggleTheme } = useTheme();
  const [isOpen, setIsOpen] = useState(false);

  const toggleMenu = () => setIsOpen(!isOpen);
  const closeMenu = () => setIsOpen(false);

  return (
    <nav className="navbar">
      <Link to="/" className="logo" onClick={closeMenu}>ArbiterLabs</Link>
      
      <div className={`links ${isOpen ? 'open' : ''}`}>
        <NavLink to="/strategies" className={({ isActive }) => isActive ? "active-link" : ""} onClick={closeMenu}>Strategies</NavLink>
        <NavLink to="/methodology" className={({ isActive }) => isActive ? "active-link" : ""} onClick={closeMenu}>Methodology</NavLink>
        <NavLink to="/logs" className={({ isActive }) => isActive ? "active-link" : ""} onClick={closeMenu}>Logs</NavLink>
        <NavLink to="/about" className={({ isActive }) => isActive ? "active-link" : ""} onClick={closeMenu}>About</NavLink>
      </div>

      <div className="nav-actions">
        <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="icon-link">
          <FaGithub />
        </a>
        <a href="https://x.com" target="_blank" rel="noopener noreferrer" className="icon-link">
          <FaTwitter />
        </a>
        <button className="theme-toggle" onClick={toggleTheme}>
          {theme === 'dark' ? <FaSun /> : <FaMoon />}
        </button>
        <button className="mobile-menu-btn" onClick={toggleMenu}>
          {isOpen ? <FaTimes /> : <FaBars />}
        </button>
      </div>
    </nav>
  );
};

export default Navbar;
