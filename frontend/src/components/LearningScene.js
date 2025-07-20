import React, { useState, useEffect } from 'react';
import ChatWindow from './ChatWindow';
import './LearningScene.css';

const LearningScene = ({ settings, onApiKeyRequired }) => {
  const [currentStudent, setCurrentStudent] = useState(null);
  const [selectedQuestion, setSelectedQuestion] = useState(null);
  const [questions, setQuestions] = useState([]);
  // 为每个题目维护独立的聊天消息状态
  const [questionMessages, setQuestionMessages] = useState({});
  const [students] = useState([
    {
      id: 'student-001',
      name: '李明',
      grade: '七年级',
      characteristics: '概念理解快，但计算粗心',
      weakPoints: ['计算准确性', '细节把控']
    },
    {
      id: 'student-002', 
      name: '王芳',
      grade: '七年级',
      characteristics: '数学基础好，但物理抽象概念理解困难',
      weakPoints: ['物理概念', '抽象思维']
    }
  ]);

  // 模拟题目数据
  useEffect(() => {
    setQuestions([
      {
        id: 'q001',
        subject: '物理',
        title: '矢量合成问题',
        content: '一个物体同时受到向东5牛和向北5牛的力，它受到的合力是多大？',
        studentAnswer: '10牛',
        correctAnswer: '5√2 牛，方向东北',
        knowledgePoints: ['矢量合成', '力的分解'],
        difficulty: '中等',
        studentId: 'student-001'
      },
      {
        id: 'q002',
        subject: '数学',
        title: '二次方程求解',
        content: '解方程：x² - 4x + 3 = 0',
        studentAnswer: 'x = 2',
        correctAnswer: 'x = 1 或 x = 3',
        knowledgePoints: ['二次方程', '因式分解'],
        difficulty: '简单',
        studentId: 'student-001'
      },
      {
        id: 'q003',
        subject: '物理',
        title: '电路分析',
        content: '在串联电路中，如果一个电阻为4Ω，另一个为6Ω，总电压为20V，求通过电路的电流。',
        studentAnswer: '2A',
        correctAnswer: '2A',
        knowledgePoints: ['串联电路', '欧姆定律'],
        difficulty: '中等',
        studentId: 'student-002'
      }
    ]);
  }, []);

  const handleStudentSelect = (student) => {
    setCurrentStudent(student);
    setSelectedQuestion(null); // 重置选中的题目
  };

  const handleQuestionSelect = (question) => {
    console.log('🎯 Question selected:', question.id, question.title);
    
    // 如果这个题目还没有消息，创建欢迎消息
    if (!questionMessages[question.id] || questionMessages[question.id].length === 0) {
      console.log('✨ Creating welcome message for question:', question.id);
      const welcomeMessage = {
        id: Date.now(),
        type: 'assistant',
        content: `你好${currentStudent.name}！我是你的AI导师。我看到你在这道${question.subject}题上遇到了一些困难。让我们一起来分析一下这道题吧！

首先，能和我说说你当时是怎么思考这道题的吗？你的解题思路是什么？别担心答错了，我们一起来找出问题所在。`,
        timestamp: new Date().toISOString()
      };
      
      setQuestionMessages(prev => ({
        ...prev,
        [question.id]: [welcomeMessage]
      }));
    }
    
    setSelectedQuestion(question);
  };

  // 获取当前题目的消息 - 只读函数，不更新状态
  const getCurrentMessages = () => {
    if (!selectedQuestion || !currentStudent) {
      console.log('❌ getCurrentMessages: No question or student selected');
      return [];
    }
    
    const messages = questionMessages[selectedQuestion.id];
    const result = Array.isArray(messages) ? messages : [];
    console.log('📋 getCurrentMessages for', selectedQuestion.id, ':', result.length, 'messages');
    
    // 确保返回的总是数组
    return result;
  };

  // 设置当前题目的消息 - 支持函数式更新
  const setCurrentMessages = (messagesOrUpdater) => {
    console.log('🔄 setCurrentMessages called for question:', selectedQuestion?.id, 'parameter type:', typeof messagesOrUpdater);
    
    if (!selectedQuestion) {
      console.log('❌ No selected question in setCurrentMessages');
      return;
    }
    
    setQuestionMessages(prev => {
      const currentMessages = prev[selectedQuestion.id] || [];
      
      let newMessages;
      if (typeof messagesOrUpdater === 'function') {
        // 函数式更新：messagesOrUpdater是一个updater函数
        newMessages = messagesOrUpdater(currentMessages);
        console.log('🔧 Function update: current', currentMessages.length, '-> new', newMessages.length);
      } else {
        // 直接设置：messagesOrUpdater是消息数组
        newMessages = Array.isArray(messagesOrUpdater) ? messagesOrUpdater : [];
        console.log('📝 Direct update:', newMessages.length, 'messages');
      }
      
      const updated = {
        ...prev,
        [selectedQuestion.id]: newMessages
      };
      console.log('📊 Updated questionMessages for', selectedQuestion.id, ':', newMessages.length, 'messages');
      return updated;
    });
  };

  const getStudentQuestions = () => {
    if (!currentStudent) return [];
    return questions.filter(q => q.studentId === currentStudent.id);
  };

  const renderStudentSelector = () => (
    <div className="student-selector">
      <h3>选择学生</h3>
      <div className="student-cards">
        {students.map(student => (
          <div 
            key={student.id}
            className={`student-card ${currentStudent?.id === student.id ? 'selected' : ''}`}
            onClick={() => handleStudentSelect(student)}
          >
            <div className="student-name">{student.name}</div>
            <div className="student-grade">{student.grade}</div>
            <div className="student-characteristics">{student.characteristics}</div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderQuestionsList = () => {
    const studentQuestions = getStudentQuestions();
    
    return (
      <div className="questions-list">
        <h3>{currentStudent.name}的错题列表</h3>
        <div className="questions-grid">
          {studentQuestions.map(question => (
            <div 
              key={question.id}
              className={`question-card ${selectedQuestion?.id === question.id ? 'selected' : ''}`}
              onClick={() => handleQuestionSelect(question)}
            >
              <div className="question-header">
                <span className="subject-tag">{question.subject}</span>
                <span className="difficulty-tag">{question.difficulty}</span>
              </div>
              <div className="question-title">{question.title}</div>
              <div className="question-preview">{question.content.slice(0, 50)}...</div>
              <div className="knowledge-points">
                {question.knowledgePoints.map(point => (
                  <span key={point} className="knowledge-point">{point}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderAITutor = () => {
    if (!selectedQuestion) return null;

    // 为AI创建专门的教育场景提示词
    const educationalPrompt = `你是一位专业的AI导师，正在为学生${currentStudent.name}提供个性化辅导。

学生特点：${currentStudent.characteristics}
学习弱点：${currentStudent.weakPoints.join('、')}

当前错题信息：
- 科目：${selectedQuestion.subject}
- 题目：${selectedQuestion.content}
- 学生答案：${selectedQuestion.studentAnswer}
- 正确答案：${selectedQuestion.correctAnswer}
- 涉及知识点：${selectedQuestion.knowledgePoints.join('、')}

请按以下步骤引导学生：
1. 先了解学生当时的思考过程
2. 诊断具体的知识点问题
3. 提供结构化的解题分析
4. 结合学生的学习特点给出针对性建议

开始时请温和地询问学生当时是怎么思考这道题的。`;

    return (
      <div className="ai-tutor-section">
        <div className="tutor-header">
          <h3>🤖 AI导师正在为 {currentStudent.name} 辅导</h3>
          <div className="question-context">
            <div className="question-info">
              <strong>{selectedQuestion.subject} - {selectedQuestion.title}</strong>
              <p>{selectedQuestion.content}</p>
              <div className="answer-comparison">
                <div className="student-answer">学生答案：{selectedQuestion.studentAnswer}</div>
                <div className="correct-answer">正确答案：{selectedQuestion.correctAnswer}</div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="chat-container">
          <ChatWindow
            settings={{
              ...settings,
              // 注入教育场景的系统提示
              educationalContext: {
                student: currentStudent,
                question: selectedQuestion,
                systemPrompt: educationalPrompt
              }
            }}
            messages={questionMessages[selectedQuestion?.id] || []}
            setMessages={setCurrentMessages}
            onApiKeyRequired={onApiKeyRequired}
            isEducationalMode={true}
          />
        </div>
      </div>
    );
  };

  return (
    <div className="learning-scene">
      {!currentStudent && renderStudentSelector()}
      
      {currentStudent && !selectedQuestion && (
        <div className="student-overview">
          <div className="back-button" onClick={() => setCurrentStudent(null)}>
            ← 返回学生选择
          </div>
          <div className="student-info">
            <h2>{currentStudent.name} ({currentStudent.grade})</h2>
            <p>学习特点：{currentStudent.characteristics}</p>
            <p>需要改进：{currentStudent.weakPoints.join('、')}</p>
          </div>
          {renderQuestionsList()}
        </div>
      )}

      {currentStudent && selectedQuestion && (
        <div className="tutoring-mode">
          <div className="back-button" onClick={() => setSelectedQuestion(null)}>
            ← 返回题目列表
          </div>
          {renderAITutor()}
        </div>
      )}
    </div>
  );
};

export default LearningScene; 