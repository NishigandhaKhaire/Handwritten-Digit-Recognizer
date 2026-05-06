import DrawingCanvas from "./components/DrawingCanvas.jsx";

function App() {
  return (
    <>
      <div className="main w-full min-h-screen flex">
        <div className="absolute flex items-center justify-center">
          <h1 className=" text-white">
            DRAW NUMBER ON CANVAS
          </h1>
        </div>
        <DrawingCanvas />
      </div>
    </>
  );
}

export default App;
