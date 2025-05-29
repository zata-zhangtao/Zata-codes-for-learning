import React from 'react';
import './App.css'; // 确保引入了样式
import TodoList from './components/TodoList'; // 确保路径正确

function App() {
  return (
    <div className="App">
      <TodoList />
    </div>
  );
}

export default App;