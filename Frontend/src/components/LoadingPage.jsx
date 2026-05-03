import { useState, useEffect } from 'react';
import { CheckCircle, Loader } from 'lucide-react';
import { useLocation, useNavigate } from 'react-router-dom';


export default function LoadingPage() {
    const location = useLocation();
    const navigate = useNavigate(); // Add this line

    const {result, patientData, modelType } = location.state || {};
    const [currentStep, setCurrentStep] = useState(0);
    const [completedSteps, setCompletedSteps] = useState([]);
    const [over, setOver] = useState([]);


    console.log

    const steps = [
        { id: 1, name: 'Pre-processing Data', duration: 1500 },
        { id: 2, name: 'Removing Null Values', duration: 1200 },
        { id: 3, name: 'Encoding Data', duration: 1800 },
        { id: 4, name: 'Preparing Dataset', duration: 1400 },
        { id: 5, name: 'Loading Model', duration: 1400 },
        { id: 6, name: 'Running Prediction', duration: 1500 }
    ];

    useEffect(() => {
        processSteps();
    }, []);

    const processSteps = async () => {
        for (let i = 0; i < steps.length; i++) {
            setCurrentStep(i);
            await new Promise(resolve => setTimeout(resolve, steps[i].duration));
            setCompletedSteps(prev => [...prev, i]);
        }

        // All steps complete, navigate to results
        navigate('/dashboard', {
            state: {
                result: result,
                patientData: patientData,  // Pass the patient data as results
                modelType: modelType
            }
        });
    };


    const getStepStatus = (index) => {
        if (completedSteps.includes(index)) return 'completed';
        if (currentStep === index) return 'processing';
        return 'pending';
    };

    return (
        
        <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50 flex items-center justify-center p-8">
            <div className="fixed top-20 left-10 w-72 h-72 bg-blue-200 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob"></div>
            <div className="fixed top-40 right-10 w-72 h-72 bg-indigo-200 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000"></div>
            <div className="fixed bottom-20 left-1/2 w-72 h-72 bg-purple-200 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000"></div>
            <div className="max-w-2xl w-full">
                {/* Header */}
                <div className="text-center mb-12">
                    <div className="w-20 h-20 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-full mx-auto mb-4 flex items-center justify-center">
                        <Loader className="w-10 h-10 text-white animate-spin" />
                    </div>
                    <h1 className="text-3xl font-bold text-gray-800 mb-2">Processing Your Request</h1>
                    <p className="text-gray-600">Please wait while we analyze the patient data...</p>
                </div>

                {/* Progress Steps */}
                <div className="bg-white rounded-xl shadow-lg p-8 border border-gray-100">
                    <div className="space-y-4">
                        {steps.map((step, index) => {
                            const status = getStepStatus(index);

                            return (
                                <div
                                    key={step.id}
                                    className={`flex items-center gap-4 p-4 rounded-lg transition-all duration-500 ${status === 'completed'
                                            ? 'bg-green-50 border-2 border-green-200'
                                            : status === 'processing'
                                                ? 'bg-blue-50 border-2 border-blue-400 shadow-md scale-105'
                                                : 'bg-gray-50 border-2 border-gray-200'
                                        }`}
                                >
                                    {/* Step Icon */}
                                    <div className="flex-shrink-0">
                                        {status === 'completed' ? (
                                            <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center">
                                                <CheckCircle className="w-6 h-6 text-white" />
                                            </div>
                                        ) : status === 'processing' ? (
                                            <div className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center">
                                                <Loader className="w-6 h-6 text-white animate-spin" />
                                            </div>
                                        ) : (
                                            <div className="w-10 h-10 bg-gray-300 rounded-full flex items-center justify-center">
                                                <span className="text-gray-600 font-semibold">{step.id}</span>
                                            </div>
                                        )}
                                    </div>

                                    {/* Step Content */}
                                    <div className="flex-1">
                                        <h3
                                            className={`font-semibold ${status === 'completed'
                                                    ? 'text-green-700'
                                                    : status === 'processing'
                                                        ? 'text-blue-700'
                                                        : 'text-gray-500'
                                                }`}
                                        >
                                            {step.name}
                                        </h3>
                                        {status === 'processing' && (
                                            <p className="text-sm text-blue-600 mt-1">Processing...</p>
                                        )}
                                        {status === 'completed' && (
                                            <p className="text-sm text-green-600 mt-1">✓ Completed</p>
                                        )}
                                    </div>

                                    {/* Status Indicator */}
                                    {status === 'processing' && (
                                        <div className="flex-shrink-0">
                                            <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse"></div>
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                    </div>

                    {/* Progress Bar */}
                    <div className="mt-8 pt-6 border-t border-gray-200">
                        <div className="flex justify-between text-sm text-gray-600 mb-2">
                            <span>Overall Progress</span>
                            <span className="font-semibold">
                                {Math.round(((completedSteps.length) / steps.length) * 50)}%
                            </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                            <div
                                className="h-full bg-gradient-to-r from-blue-600 to-indigo-600 rounded-full transition-all duration-500 ease-out"
                                style={{ width: `${((completedSteps.length) / steps.length) * 50}%` }}
                            ></div>
                        </div>
                    </div>
                </div>

                {/* Patient Info */}
                <div className="mt-6 bg-white rounded-xl shadow-lg p-6 border border-gray-100">
                    <h3 className="font-semibold text-gray-700 mb-3 text-sm uppercase tracking-wide">
                        Processing Details
                    </h3>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                            <span className="text-gray-600">Patient ID:</span>
                            <p className="font-semibold text-gray-800">{patientData?.patient_practice_id || 'N/A'}</p>
                        </div>
                        <div>
                            <span className="text-gray-600">Model:</span>
                            <p className="font-semibold text-gray-800">
                                {modelType?.replace('_', ' ').toUpperCase() || 'N/A'}
                            </p>
                        </div>
                    </div>
                </div>
            </div>
            
        </div>
    );
}