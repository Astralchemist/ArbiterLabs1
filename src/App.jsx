import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import Navbar from './components/Navbar';
import Disclaimer from './components/Disclaimer';
import Home from './pages/Home';
import StrategiesPage from './pages/StrategiesPage';
import LogsPage from './pages/LogsPage';
import MethodologyPage from './pages/MethodologyPage';
import AboutPage from './pages/AboutPage';
import './index.css';

function App() {
  return (
    <ThemeProvider>
      <Router>
        <div className="app-container">
          <Navbar />
          <main>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/strategies" element={<StrategiesPage />} />
              <Route path="/logs" element={<LogsPage />} />
              <Route path="/methodology" element={<MethodologyPage />} />
              <Route path="/about" element={<AboutPage />} />
            </Routes>
            <Disclaimer />
          </main>
        </div>
      </Router>
    </ThemeProvider>
  );
}

export default App;
