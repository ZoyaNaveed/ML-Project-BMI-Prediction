import { useState } from 'react';
import { Activity, Target, TrendingUp, Shield, Zap, Users, ChevronRight, Check } from 'lucide-react';
import {useNavigate} from 'react-router-dom'

export default function LandingPage() {


    const features = [
        {
            icon: <Activity className="w-6 h-6" />,
            title: "Accurate Predictions",
            description: "Advanced AI algorithms provide precise BMI calculations and health insights"
        },
        {
            icon: <Target className="w-6 h-6" />,
            title: "Personalized Goals",
            description: "Get custom health targets based on your unique body composition"
        },
        {
            icon: <TrendingUp className="w-6 h-6" />,
            title: "Progress Tracking",
            description: "Monitor your journey with detailed analytics and visual reports"
        },
        {
            icon: <Shield className="w-6 h-6" />,
            title: "Privacy First",
            description: "Your health data is encrypted and never shared with third parties"
        }
    ];

    const plans = [
        {
            name: "Free",
            price: "$0",
            period: "forever",
            features: [
                "Basic BMI Calculator",
                "Health Category Classification",
                "Weekly Tips",
                "Community Access"
            ]
        },
        {
            name: "Pro",
            price: "$9.99",
            period: "per month",
            popular: true,
            features: [
                "Everything in Free",
                "Advanced Body Composition Analysis",
                "Personalized Meal Plans",
                "Fitness Recommendations",
                "Priority Support",
                "Export Reports"
            ]
        },
        {
            name: "Enterprise",
            price: "Custom",
            period: "contact us",
            features: [
                "Everything in Pro",
                "Team Management",
                "API Access",
                "White-label Solution",
                "Dedicated Account Manager",
                "Custom Integrations"
            ]
        }
    ];

    const stats = [
        { value: "500K+", label: "Active Users" },
        { value: "98%", label: "Accuracy Rate" },
        { value: "4.9/5", label: "User Rating" },
        { value: "50+", label: "Countries" }
    ];
    const navigate = useNavigate();

    const handleGetStarted = () => {
        navigate('/patients-page');
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50 bg-gray-50">
            {/* Navigation */}


            {/* Hero Section */}
            <section className="w-full bg-gradient-to-br">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 lg:py-20">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-16 items-center">
                        <div className="space-y-8">
                            <div className="inline-flex items-center gap-2 bg-blue-100 text-blue-700 px-4 py-2 rounded-full text-sm font-medium">
                                <Zap className="w-4 h-4" />
                                <span>AI-Powered Health Intelligence</span>
                            </div>
                            <h1 className="text-5xl md:text-6xl font-bold leading-tight">
                                Transform Your Health Journey with{' '}
                                <span className="text-blue-600">
                                    Smart BMI Tracking
                                </span>
                            </h1>
                            <p className="text-xl text-gray-600 leading-relaxed">
                                Unlock personalized health insights with our advanced BMI prediction system.
                                Track, analyze, and achieve your wellness goals with data-driven precision.
                            </p>
                            <div className="flex flex-col sm:flex-row gap-4">

                                <button
                                    onClick={() => navigate('/patients-page')}
                                    className="bg-blue-600 text-white px-8 py-4 rounded-lg font-semibold hover:shadow-xl transition flex items-center justify-center gap-2 group">

                                    Get Started

                                    <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition" />
                                </button>
                            </div>

                        </div>
                        <div className="relative w-full max-w-md mx-auto lg:max-w-none">
                            <div className="bg-blue-600 rounded-3xl p-8 shadow-2xl transform hover:scale-105 transition duration-500">
                                <div className="bg-white rounded-2xl p-8 space-y-6">
                                    <div className="flex items-center justify-between">
                                        <h3 className="text-2xl font-bold text-gray-800">Your Health Score</h3>
                                        <div className="bg-green-100 text-green-700 px-4 py-2 rounded-full text-sm font-semibold">
                                            Healthy
                                        </div>
                                    </div>
                                    <div className="relative pt-4">
                                        <div className="flex justify-center">
                                            <div className="relative w-48 h-48">
                                                <svg className="transform -rotate-90 w-48 h-48">
                                                    <circle cx="96" cy="96" r="88" stroke="#e5e7eb" strokeWidth="12" fill="none" />
                                                    <circle cx="96" cy="96" r="88" stroke="#2563eb" strokeWidth="12" fill="none" strokeDasharray="552.92" strokeDashoffset="138.23" strokeLinecap="round" />
                                                </svg>
                                                <div className="absolute inset-0 flex items-center justify-center flex-col">
                                                    <span className="text-4xl font-bold text-blue-600">22.5</span>
                                                    <span className="text-gray-500 text-sm">BMI Score</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="grid grid-cols-3 gap-4 pt-4">
                                        <div className="text-center">
                                            <div className="text-2xl font-bold text-gray-800">68</div>
                                            <div className="text-xs text-gray-500">Weight (kg)</div>
                                        </div>
                                        <div className="text-center">
                                            <div className="text-2xl font-bold text-gray-800">175</div>
                                            <div className="text-xs text-gray-500">Height (cm)</div>
                                        </div>
                                        <div className="text-center">
                                            <div className="text-2xl font-bold text-gray-800">25</div>
                                            <div className="text-xs text-gray-500">Age</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div className="absolute -bottom-6 -right-6 bg-white p-6 rounded-2xl shadow-xl">
                                <div className="flex items-center gap-3">
                                    <Users className="w-8 h-8 text-blue-600" />
                                    <div>
                                        <div className="text-2xl font-bold text-gray-800">500K+</div>
                                        <div className="text-sm text-gray-500">Happy Users</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Stats Section */}
            <section className="bg-blue-600 py-16">
                <div className="max-w-7xl mx-auto px-6">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
                        {stats.map((stat, index) => (
                            <div key={index} className="text-center">
                                <div className="text-4xl md:text-5xl font-bold text-white mb-2">{stat.value}</div>
                                <div className="text-blue-100">{stat.label}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Features Section */}
            <section id="features" className="max-w-7xl mx-auto px-6 py-20">
                <div className="text-center mb-16">
                    <h2 className="text-4xl md:text-5xl font-bold mb-4">
                        Powerful Features for Your{' '}
                        <span className="text-blue-600">
                            Health Success
                        </span>
                    </h2>
                    <p className="text-xl text-gray-600 max-w-2xl mx-auto">
                        Everything you need to monitor, analyze, and improve your health in one intelligent platform
                    </p>
                </div>
                <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
                    {features.map((feature, index) => (
                        <div key={index} className="bg-white p-8 rounded-2xl shadow-lg hover:shadow-2xl transition group cursor-pointer">
                            <div className="bg-blue-100 w-14 h-14 rounded-xl flex items-center justify-center text-blue-600 mb-6 group-hover:scale-110 transition">
                                {feature.icon}
                            </div>
                            <h3 className="text-xl font-bold mb-3 text-gray-800">{feature.title}</h3>
                            <p className="text-gray-600 leading-relaxed">{feature.description}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* CTA Section */}
            <section className="max-w-7xl mx-auto px-6 py-20">
                <div className="bg-blue-600 rounded-3xl p-12 md:p-16 text-center text-white">
                    <h2 className="text-4xl md:text-5xl font-bold mb-6">
                        Ready to Transform Your Health?
                    </h2>
                    <p className="text-xl mb-8 text-blue-100 max-w-2xl mx-auto">
                        Join thousands of users who are already achieving their health goals with BMI Predictor
                    </p>
                    <button className="bg-white text-blue-600 px-8 py-4 rounded-lg font-semibold text-lg hover:shadow-2xl transition inline-flex items-center gap-2 group">
                        Get Started For free!

                        <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition" />
                    </button>
                </div>
            </section>

            {/* Footer */}
            <footer className="bg-gray-900 text-gray-400 py-12">
                <div className="max-w-7xl mx-auto px-6">
                    <div className="grid md:grid-cols-4 gap-8 mb-8">
                        <div>
                            <div className="flex items-center gap-2 mb-4">
                                <div className="bg-blue-600 p-2 rounded-lg">
                                    <Activity className="w-5 h-5 text-white" />
                                </div>
                                <span className="text-white font-bold">BMI Predictor</span>
                            </div>
                            <p className="text-sm">Empowering healthier lives through intelligent BMI tracking and personalized insights.</p>
                        </div>
                        <div>
                            <h4 className="text-white font-semibold mb-4">Product</h4>
                            <ul className="space-y-2 text-sm">
                                <li><a href="#" className="hover:text-white transition">Features</a></li>
                                <li><a href="#" className="hover:text-white transition">Pricing</a></li>
                                <li><a href="#" className="hover:text-white transition">API</a></li>
                            </ul>
                        </div>
                        <div>
                            <h4 className="text-white font-semibold mb-4">Company</h4>
                            <ul className="space-y-2 text-sm">
                                <li><a href="#" className="hover:text-white transition">About</a></li>
                                <li><a href="#" className="hover:text-white transition">Blog</a></li>
                                <li><a href="#" className="hover:text-white transition">Careers</a></li>
                            </ul>
                        </div>
                        <div>
                            <h4 className="text-white font-semibold mb-4">Legal</h4>
                            <ul className="space-y-2 text-sm">
                                <li><a href="#" className="hover:text-white transition">Privacy</a></li>
                                <li><a href="#" className="hover:text-white transition">Terms</a></li>
                                <li><a href="#" className="hover:text-white transition">Security</a></li>
                            </ul>
                        </div>
                    </div>
                    <div className="border-t border-gray-800 pt-8 text-center text-sm">
                        <p>&copy; 2025 BMI Predictor. All rights reserved.</p>
                    </div>
                </div>
            </footer>
            
        </div>
    );
}