import { BrowserRouter, Route, Routes } from 'react-router-dom'
import LandingPage from './components/LandingPage'
import PatientsPage from './components/PatientsPage'
import LoadingPage from './components/LoadingPage'
import Dashboard from './components/Dashboard'
import Navbar from './components/Navbar'

function App() {
  return (


    <BrowserRouter>
      {/* ✅ Navbar should be outside Routes so it shows on all pages */}
      <Navbar />

      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/patients-page" element={<PatientsPage />} />
        <Route path="/loading-page" element={<LoadingPage />} />
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
      </BrowserRouter>
  )
}

export default App
