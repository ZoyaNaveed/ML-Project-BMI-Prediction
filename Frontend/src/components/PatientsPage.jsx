import { useState, useEffect } from 'react';
import Papa from 'papaparse';
import { useNavigate } from 'react-router-dom';
import { Activity, CheckCircle, Users, Brain, Zap } from 'lucide-react';

export default function MinimalPatientSelector() {
    const [patients, setPatients] = useState([]);
    const [selectedModel, setSelectedModel] = useState('');
    const [selectedPatientId, setSelectedPatientId] = useState('');
    const [currentPage, setCurrentPage] = useState(0);
    const [loading, setLoading] = useState(false);

    const patientsPerPage = 10;
    const models = [
        { id: 'XGBoost', name: 'XGBoost', icon: '⚡' },
        { id: 'Ridge_Regression', name: 'Ridge Regression', icon: '📊', },
        { id: 'Lasso_Regression', name: 'Lasso Regression', icon: '📈',  },
        { id: 'ElasticNet_Regression', name: 'ElasticNet', icon: '🎯', },
        { id: 'Random_Forest', name: 'Random Forest', icon: '🌲',  },
        { id: 'Gradient_Boosting', name: 'Gradient Boosting', icon: '🚀', }
    ];

    const navigate = useNavigate();

    useEffect(() => {
        loadCSV();
    }, []);

    const loadCSV = async () => {
        setLoading(true);
        try {
            const response = await fetch('/Patients_holdout.csv');
            const csvText = await response.text();

            const result = Papa.parse(csvText, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                transformHeader: (h) => h.trim()
            });

            console.log('Loaded patients:', result.data.length);
            setPatients(result.data);
        } catch (error) {
            console.error('Error loading CSV:', error);
            alert('Failed to load CSV file');
        }
        setLoading(false);
    };

    const getCurrentPatients = () => {
        const start = currentPage * patientsPerPage;
        return patients.slice(start, start + patientsPerPage);
    };

    const handleSubmit = async () => {
        if (!selectedModel || !selectedPatientId) {
            alert('Please select both a model and a patient');
            return;
        }

        const patientData = patients.find(
            p => p.patient_practice_id === selectedPatientId
        );

        if (!patientData) {
            alert('Patient data not found');
            return;
        }

        const payload = {
            model_name: selectedModel,
            patient_data: patientData
        };

        try {
            const response = await fetch('http://localhost:5000/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            navigate('/loading-page', {
                state: {
                    patientData: patientData,
                    result: result,
                    modelType: selectedModel
                }
            });

        } catch (error) {
            console.error('API Error:', error);
            alert('API request failed - check console for payload');
            console.log('Payload that would be sent:', JSON.stringify(payload, null, 2));
        }
    };

    const totalPages = Math.ceil(patients.length / patientsPerPage);

    if (loading) {
        return (
            <div className="min-h-screen  flex items-center justify-center">
                <div className="text-center">
                    <div className="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                    <p className="text-gray-700 font-medium">Loading patients...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50 p-8">
            {/* Decorative elements */}
            <div className="fixed top-20 left-10 w-72 h-72 bg-blue-200 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob"></div>
            <div className="fixed top-40 right-10 w-72 h-72 bg-indigo-200 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000"></div>
            <div className="fixed bottom-20 left-1/2 w-72 h-72 bg-purple-200 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000"></div>

            <div className="max-w-5xl mx-auto relative">
                {/* Header */}
                <div className="text-center mb-12">
                    <div className="flex items-center justify-center gap-3 mb-4">
                        <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg">
                            <Activity className="w-7 h-7 text-white" />
                        </div>
                        <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                            BMI Prediction
                        </h1>
                    </div>
                    <p className="text-gray-600 text-lg">Select your model and patient to begin analysis</p>
                    <div className="mt-6 inline-flex items-center gap-2 bg-blue-100 text-blue-700 px-4 py-2 rounded-full text-sm font-medium">
                        <Users className="w-4 h-4" />
                        <span>{patients.length} patients available</span>
                    </div>
                </div>

                {/* Progress Steps */}
                <div className="flex items-center justify-center mb-12">
                    <div className="flex items-center gap-4">
                        <div className={`flex items-center gap-2 px-4 py-2 rounded-full ${selectedModel ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'}`}>
                            <div className={`w-6 h-6 rounded-full flex items-center justify-center font-bold text-sm ${selectedModel ? 'bg-green-600 text-white' : 'bg-blue-600 text-white'}`}>
                                {selectedModel ? '✓' : '1'}
                            </div>
                            <span className="text-sm font-semibold">Select Model</span>
                        </div>
                        <div className="w-12 h-0.5 bg-gray-300"></div>
                        <div className={`flex items-center gap-2 px-4 py-2 rounded-full ${selectedPatientId ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'}`}>
                            <div className={`w-6 h-6 rounded-full flex items-center justify-center font-bold text-sm ${selectedPatientId ? 'bg-green-600 text-white' : 'bg-gray-400 text-white'}`}>
                                {selectedPatientId ? '✓' : '2'}
                            </div>
                            <span className="text-sm font-semibold">Select Patient</span>
                        </div>
                    </div>
                </div>

                {/* Model Selection */}
                <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl p-8 mb-8 border border-gray-200">
                    <div className="flex items-center gap-3 mb-6">
                        <Brain className="w-6 h-6 text-blue-600" />
                        <h2 className="text-2xl font-bold text-gray-800">Choose Prediction Model</h2>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {models.map((model) => (
                            <button
                                key={model.id}
                                onClick={() => setSelectedModel(model.id)}
                                className={`group relative p-5 rounded-xl border-2 transition-all duration-300 ${selectedModel === model.id
                                        ? 'border-blue-600 bg-gradient-to-br from-blue-50 to-indigo-50 shadow-xl scale-105'
                                        : 'border-gray-200 bg-white hover:border-blue-300 hover:shadow-lg hover:scale-102'
                                    }`}
                            >
                                <div className="flex items-start justify-between mb-3">
                                    <span className="text-3xl">{model.icon}</span>
                                    {selectedModel === model.id && (
                                        <CheckCircle className="w-6 h-6 text-blue-600" />
                                    )}
                                </div>
                                <div className="text-left">
                                    <div className="font-bold text-gray-800 mb-1">{model.name}</div>
                                </div>
                            </button>
                        ))}
                    </div>
                </div>

                {/* Patient Selection */}
                <div className={`bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl p-8 mb-8 border border-gray-200 transition-all duration-500 ${!selectedModel ? 'opacity-50' : 'opacity-100'}`}>
                    <div className="flex items-center justify-between mb-6">
                        <div className="flex items-center gap-3">
                            <Users className="w-6 h-6 text-blue-600" />
                            <h2 className="text-2xl font-bold text-gray-800">Select Patient</h2>
                        </div>
                        <span className="text-sm text-gray-500 bg-gray-100 px-4 py-2 rounded-full font-semibold">
                            Page {currentPage + 1} of {totalPages}
                        </span>
                    </div>

                    <div className="space-y-3 mb-6">
                        {getCurrentPatients().map((patient, index) => (
                            <button
                                key={index}
                                onClick={() => selectedModel && setSelectedPatientId(patient.patient_practice_id)}
                                disabled={!selectedModel}
                                className={`w-full p-5 rounded-xl border-2 text-left transition-all duration-300 ${selectedPatientId === patient.patient_practice_id
                                        ? 'border-blue-600 bg-gradient-to-r from-blue-50 to-indigo-50 shadow-lg scale-102'
                                        : 'border-gray-200 bg-white hover:border-blue-300 hover:shadow-md'
                                    } ${!selectedModel ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}`}
                            >
                                <div className="flex justify-between items-center">
                                    <div className="flex items-center gap-4">
                                        <div className="w-12 h-12 bg-gradient-to-br from-blue-100 to-indigo-100 rounded-full flex items-center justify-center">
                                            <Users className="w-6 h-6 text-blue-600" />
                                        </div>
                                        <div>
                                            <span className="font-bold text-gray-800 text-lg block">
                                                {patient.patient_practice_id}
                                            </span>
                                            <span className="text-xs text-gray-500">Click to select</span>
                                        </div>
                                    </div>
                                    {selectedPatientId === patient.patient_practice_id && (
                                        <div className="bg-blue-600 text-white px-4 py-2 rounded-full font-semibold text-sm flex items-center gap-2">
                                            <CheckCircle className="w-4 h-4" />
                                            Selected
                                        </div>
                                    )}
                                </div>
                            </button>
                        ))}
                    </div>

                    {/* Pagination */}
                    <div className="flex justify-between items-center pt-6 border-t border-gray-200">
                        <button
                            onClick={() => setCurrentPage(p => Math.max(0, p - 1))}
                            disabled={currentPage === 0}
                            className="px-6 py-3 bg-blue-600 text-white rounded-xl font-semibold disabled:bg-gray-300 disabled:cursor-not-allowed hover:bg-blue-700 transition-all shadow-md hover:shadow-lg"
                        >
                            ← Previous
                        </button>
                        <div className="text-gray-600 font-medium">
                            Showing {currentPage * patientsPerPage + 1}-{Math.min((currentPage + 1) * patientsPerPage, patients.length)} of {patients.length}
                        </div>
                        <button
                            onClick={() => setCurrentPage(p => Math.min(totalPages - 1, p + 1))}
                            disabled={currentPage >= totalPages - 1}
                            className="px-6 py-3 bg-blue-600 text-white rounded-xl font-semibold disabled:bg-gray-300 disabled:cursor-not-allowed hover:bg-blue-700 transition-all shadow-md hover:shadow-lg"
                        >
                            Next →
                        </button>
                    </div>
                </div>

                {/* Submit Button */}
                <button
                    onClick={handleSubmit}
                    disabled={!selectedModel || !selectedPatientId}
                    className="w-full py-6 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-2xl font-bold text-xl disabled:from-gray-300 disabled:to-gray-400 disabled:cursor-not-allowed hover:from-blue-700 hover:to-indigo-700 transition-all shadow-xl hover:shadow-2xl transform hover:scale-102 active:scale-98 flex items-center justify-center gap-3"
                >
                    <Zap className="w-6 h-6" />
                    {!selectedModel || !selectedPatientId ? 'Complete Selection to Continue' : 'Generate Prediction'}
                </button>

                {/* Summary Card */}
                {(selectedModel || selectedPatientId) && (
                    <div className="mt-8 bg-gradient-to-r from-gray-50 to-blue-50 rounded-2xl p-6 border border-gray-200 shadow-lg">
                        <h3 className="font-bold text-gray-700 mb-4 text-sm uppercase tracking-wide flex items-center gap-2">
                            <CheckCircle className="w-4 h-4 text-blue-600" />
                            Current Selection
                        </h3>
                        <div className="grid grid-cols-2 gap-4">
                            <div className="bg-white rounded-lg p-4">
                                <div className="text-xs text-gray-500 mb-1">Model</div>
                                <div className="font-bold text-gray-800">
                                    {selectedModel ? models.find(m => m.id === selectedModel)?.name : '—'}
                                </div>
                            </div>
                            <div className="bg-white rounded-lg p-4">
                                <div className="text-xs text-gray-500 mb-1">Patient ID</div>
                                <div className="font-bold text-gray-800 truncate">
                                    {selectedPatientId || '—'}
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>

        </div>
    );
}