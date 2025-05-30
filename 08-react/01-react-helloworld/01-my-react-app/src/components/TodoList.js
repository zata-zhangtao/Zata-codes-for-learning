import React, { useState } from 'react';
import TodoItem from './TodoItem';

function TodoList() {
  const [todos, setTodos] = useState([
    { text: '学习 React', isCompleted: false },
    { text: '构建一个 Todo 应用', isCompleted: false },
  ]);
  const [inputValue, setInputValue] = useState('');

  const addTodo = (text) => {
    if (!text.trim()) return; // 不添加空内容
    const newTodos = [...todos, { text, isCompleted: false }];
    setTodos(newTodos);
  };

  const toggleTodo = (index) => {
    const newTodos = [...todos];
    newTodos[index].isCompleted = !newTodos[index].isCompleted;
    setTodos(newTodos);
  };

  const removeTodo = (index) => {
    const newTodos = [...todos];
    newTodos.splice(index, 1);
    setTodos(newTodos);
  };

  const handleSubmit = (e) => {
    e.preventDefault(); // 阻止表单默认提交行为
    addTodo(inputValue);
    setInputValue(''); // 清空输入框
  };

  return (
    <div className="todo-list">
      <h1>我的待办事项</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="添加新的待办..."
        />
        <button type="submit">添加</button>
      </form>
      <ul>
        {todos.map((todo, index) => (
          <TodoItem
            key={index} // 注意：这里用 index 做 key 仅为演示，实际项目中如有唯一 ID 更好
            index={index}
            todo={todo}
            toggleTodo={toggleTodo}
            removeTodo={removeTodo}
          />
        ))}
      </ul>
    </div>
  );
}

export default TodoList;