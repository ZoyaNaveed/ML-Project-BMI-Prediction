// src/components/Navbar.jsx
import { Activity } from "lucide-react";
import {useNavigate} from 'react-router-dom'

export default function Navbar() {
    const navigate = useNavigate();

    const handleGetStarted = () => {
        navigate('/patients-page');
    };
    return (
        <nav className="bg-white/80 backdrop-blur-md border-b border-gray-200 sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-6 py-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <div className="bg-blue-600 p-2 rounded-lg">
                            <Activity className="w-6 h-6 text-white" />
                        </div>
                        <span className="text-xl font-bold text-blue-600">BMI Predictor</span>
                    </div>
                    <div className="hidden md:flex items-center gap-8">
                        {/* Use NavLink for active styles */}
                        {/* <NavLink> requires react-router-dom */}
                        {/* Replace hrefs with to=... */}
                        {/* Example: <NavLink to="/features" className=...>Features</NavLink> */}
                        <a href="#features" className="text-gray-600 hover:text-blue-600 transition">Features</a>
                        {/* <a href="#about" className="text-gray-600 hover:text-blue-600 transition">About</a> */}
                        {/* <a href="/signup" className="text-gray-600 hover:text-blue-600 transition">Sign Up</a> */}
                        <button className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:shadow-lg transition"
                            onClick={() => navigate('/')}>
                            
                            Get Started
                        </button>
                    </div>
                </div>
            </div>
        </nav>
    );
}
