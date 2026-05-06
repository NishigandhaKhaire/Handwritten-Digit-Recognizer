import React, { useRef, useState, useEffect } from 'react';
import axios from 'axios';

const DrawingCanvas = () => {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [predictions, setPredictions] = useState([]);
  let xCoords = [];
  let yCoords = [];

  const digitToWord = {
    0: 'zero',
    1: 'one',
    2: 'two',
    3: 'three',
    4: 'four',
    5: 'five',
    6: 'six',
    7: 'seven',
    8: 'eight',
    9: 'nine',
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const handleResize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize); 
    };
  }, []);

  const startDrawing = ({ nativeEvent }) => {
    const { offsetX, offsetY } = nativeEvent;
    const context = canvasRef.current.getContext('2d');
    context.beginPath();
    context.moveTo(offsetX, offsetY);
    setIsDrawing(true);
    xCoords.push(offsetX);
    yCoords.push(offsetY);
  };

  const draw = ({ nativeEvent }) => {
    if (!isDrawing) return;
    const { offsetX, offsetY } = nativeEvent;
    const context = canvasRef.current.getContext('2d');
    context.lineTo(offsetX, offsetY);
    context.strokeStyle = 'white';
    context.lineWidth = 4;
    context.stroke();
    xCoords.push(offsetX);
    yCoords.push(offsetY);
  };

  const endDrawing = async () => {
    setIsDrawing(false);
    if (xCoords.length === 0 || yCoords.length === 0) return;

    const minX = Math.min(...xCoords) - 10;
    const minY = Math.min(...yCoords) - 10;
    const maxX = Math.max(...xCoords) + 10;
    const maxY = Math.max(...yCoords) + 10;

    xCoords = [];
    yCoords = [];

    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    const imageData = context.getImageData(minX, minY, maxX - minX, maxY - minY);

    const tempCanvas = document.createElement('canvas');
    const tempContext = tempCanvas.getContext('2d');
    tempCanvas.width = imageData.width;
    tempCanvas.height = imageData.height;
    tempContext.putImageData(imageData, 0, 0);

    const base64Image = tempCanvas.toDataURL('image/png');

    try {
      
      const response = await axios.post(`${import.meta.env.VITE_BACKEND_URL}/predict`, {
        image: base64Image,
      });
      const result = response.data;

      context.strokeStyle = 'red';
      context.lineWidth = 2;
      context.strokeRect(minX, minY, maxX - minX, maxY - minY);

      setPredictions((prev) => [
        ...prev,
        {
          digit: digitToWord[result.digit],
          x: minX,
          y: minY,
        },
      ]);
    } catch (error) {
      console.error('Error predicting the digit:', error);
    }
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);
    setPredictions([]);
  };

  return (
    <div>
      <canvas
        ref={canvasRef}
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={endDrawing}
        style={{ backgroundColor: 'black' }}
      />
      
      {predictions.map((prediction, index) => (
        <div
          key={index}
          style={{
            position: 'absolute',
            left: `${prediction.x}px`,
            top: `${prediction.y - 20}px`,
            color: 'red',
            backgroundColor: 'white',
            padding: '2px',
          }}
        >
          {prediction.digit}
        </div>
      ))}
      <button
        onClick={clearCanvas}
        style={{
          position: 'absolute',
          bottom: '20px',
          right: '20px',
          padding: '10px 20px',
          backgroundColor: '#ff4c4c',
          color: 'white',
          border: 'none',
          borderRadius: '5px',
          cursor: 'pointer',
          fontSize: '16px'
        }}
      >
        Clear
      </button>
    </div>
  );
};

export default DrawingCanvas;
