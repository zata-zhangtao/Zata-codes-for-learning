import React from 'react';

function TodoItem({ todo, index, toggleTodo, removeTodo }) {
  return (
    <li style={{ textDecoration: todo.isCompleted ? 'line-through' : '' }}>
      {todo.text}
      <div>
        <button onClick={() => toggleTodo(index)}>
          {todo.isCompleted ? '撤销' : '完成'}
        </button>
        <button onClick={() => removeTodo(index)} style={{ marginLeft: '10px' }}>
          删除
        </button>
      </div>
    </li>
  );
}

export default TodoItem;