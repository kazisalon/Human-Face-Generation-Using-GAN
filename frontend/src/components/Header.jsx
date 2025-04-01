// src/components/Header.jsx
function Header() {
    return (
      <header className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white py-12">
        <div className="container mx-auto px-4 text-center">
          <h1 className="text-4xl font-bold mb-2">Human Face Generator</h1>
          <p className="text-xl opacity-90">Create realistic human faces using Generative Adversarial Networks</p>
        </div>
      </header>
    );
  }
  
  export default Header;