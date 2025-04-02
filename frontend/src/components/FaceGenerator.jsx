function FaceGenerator({ seed, onSeedChange, onGenerate, loading }) {
    return (
      <div>
        <h2 className="text-2xl font-bold mb-4">Generate a Face</h2>
        
        <div className="mb-4">
          <label htmlFor="seed" className="block text-gray-700 mb-2">
            Seed (optional)
          </label>
          <input
            type="number"
            id="seed"
            className="w-full p-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Enter a seed for reproducible results"
            value={seed}
            onChange={onSeedChange}
          />
          <p className="text-sm text-gray-500 mt-1">
            Leave empty for random faces, or enter a number to generate consistent results
          </p>
        </div>
        
        <button
          onClick={onGenerate}
          disabled={loading}
          className={`w-full py-3 px-4 rounded font-medium ${
            loading
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700 text-white transition-colors'
          }`}
        >
          {loading ? 'Generating...' : 'Generate Face'}
        </button>
      </div>
    );
  }
  
  export default FaceGenerator;