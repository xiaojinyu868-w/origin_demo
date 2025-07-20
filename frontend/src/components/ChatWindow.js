import React, { useState, useRef, useEffect } from 'react';
import ChatBubble from './ChatBubble';
import MessageInput from './MessageInput';
import ApiKeyModal from './ApiKeyModal';
import ClearChatModal from './ClearChatModal';
import queuedFetch from '../utils/requestQueue';
import './ChatWindow.css';

const ChatWindow = ({ 
  settings, 
  messages, 
  setMessages, 
  onApiKeyRequired,
  isEducationalMode = false 
}) => {
  const [includeScreenshots, setIncludeScreenshots] = useState(true);
  const [currentModel, setCurrentModel] = useState(settings.model); // Track actual current model
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  const [missingApiKeys, setMissingApiKeys] = useState([]);
  const [currentModelType, setCurrentModelType] = useState('');
  // Track active streaming requests
  const [activeStreamingRequests, setActiveStreamingRequests] = useState(new Map());
  // Clear chat modal state
  const [showClearModal, setShowClearModal] = useState(false);
  const [isClearing, setIsClearing] = useState(false);
  const messagesEndRef = useRef(null);
  const abortControllersRef = useRef(new Map());

  // Calculate derived values from state early
  const hasActiveStreaming = activeStreamingRequests.size > 0;
  const currentStreamingContent = hasActiveStreaming 
    ? Array.from(activeStreamingRequests.values())[activeStreamingRequests.size - 1].streamingContent
    : '';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, currentStreamingContent]);

  useEffect(() => {
    return () => {
      // Cleanup all abort controllers on unmount
      abortControllersRef.current.forEach((controller) => {
        controller.abort();
      });
      abortControllersRef.current.clear();
    };
  }, []);

  // Load initial screenshot setting
  useEffect(() => {
    const loadScreenshotSetting = async () => {
      try {
        const response = await queuedFetch(`${settings.serverUrl}/screenshot_setting`);
        if (response.ok) {
          const data = await response.json();
          setIncludeScreenshots(data.include_recent_screenshots);
        }
      } catch (error) {
        console.error('Error loading screenshot setting:', error);
      }
    };
    
    loadScreenshotSetting();
  }, [settings.serverUrl]);

  // Load current model
  useEffect(() => {
    const loadCurrentModel = async () => {
      try {
        const response = await queuedFetch(`${settings.serverUrl}/models/current`);
        if (response.ok) {
          const data = await response.json();
          setCurrentModel(data.current_model);
        }
      } catch (error) {
        console.error('Error loading current model:', error);
        // Fallback to settings.model if API call fails
        setCurrentModel(settings.model);
      }
    };
    
    loadCurrentModel();
  }, [settings.serverUrl, settings.model]);

  // Refresh data when backend reconnects
  useEffect(() => {
    const refreshBackendData = async () => {
      if (settings.lastBackendRefresh && settings.serverUrl) {
        console.log('ChatWindow: backend reconnected, refreshing data');
        
        // Reload screenshot setting
        try {
          const response = await queuedFetch(`${settings.serverUrl}/screenshot_setting`);
          if (response.ok) {
            const data = await response.json();
            setIncludeScreenshots(data.include_recent_screenshots);
          }
        } catch (error) {
          console.error('Error reloading screenshot setting:', error);
        }
        
        // Reload current model
        try {
          const response = await queuedFetch(`${settings.serverUrl}/models/current`);
          if (response.ok) {
            const data = await response.json();
            setCurrentModel(data.current_model);
          }
        } catch (error) {
          console.error('Error reloading current model:', error);
          // Fallback to settings.model if API call fails
          setCurrentModel(settings.model);
        }
      }
    };
    
    refreshBackendData();
  }, [settings.lastBackendRefresh, settings.serverUrl, settings.model]);

  // Function to save image files to local directory
  const saveImageToLocal = async (file) => {
    // Check if we're in Electron environment and handlers are available
    const isElectronWithHandlers = window.electronAPI && 
      typeof window.electronAPI.saveImageToTmp === 'function' &&
      typeof window.electronAPI.saveImageBufferToTmp === 'function';

    if (!isElectronWithHandlers) {
      // For web environment or Electron without handlers, handle files appropriately
      console.log('Running in web mode or Electron handlers not ready, using web fallback');
      
      if (file.file) {
        // For File objects in web environment, convert to base64 data URL
        try {
          const base64 = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file.file);
          });
          
          return {
            name: file.name,
            path: file.name, // Keep original filename for backend reference
            displayUrl: base64, // Use base64 data URL for display
            type: file.type,
            size: file.size,
            isScreenshot: file.isScreenshot || false,
            isBase64: true // Flag to indicate this is base64 data
          };
        } catch (error) {
          console.error('Error converting file to base64:', error);
          return {
            name: file.name,
            path: file.name,
            type: file.type,
            size: file.size,
            isScreenshot: file.isScreenshot || false,
            error: 'Failed to process file'
          };
        }
      }
      
      // For other file types or screenshots, return as-is with safe fallback
      return {
        name: file.name,
        path: file.path || file.name,
        type: file.type,
        size: file.size,
        isScreenshot: file.isScreenshot || false,
        ...(file.lastModified && { lastModified: file.lastModified })
      };
    }

    // Electron environment with handlers available
    try {
      // Generate unique filename
      const timestamp = Date.now();
      const randomId = Math.random().toString(36).substr(2, 9);
      const extension = file.name.split('.').pop() || 'png';
      const uniqueFileName = `${timestamp}_${randomId}.${extension}`;
      
      // For screenshots, the file.path is already the full path
      if (file.isScreenshot && file.path) {
        const savedPath = await window.electronAPI.saveImageToTmp(file.path, uniqueFileName);
        
        // Also get base64 for display purposes (to avoid file:// security issues)
        let displayUrl = null;
        try {
          if (window.electronAPI.readImageAsBase64) {
            const base64Result = await window.electronAPI.readImageAsBase64(savedPath);
            if (base64Result.success) {
              displayUrl = base64Result.dataUrl;
            }
          }
        } catch (error) {
          console.warn('Could not read saved screenshot as base64:', error);
        }
        
        return {
          name: file.name,
          path: savedPath, // File path for backend
          displayUrl: displayUrl, // Base64 URL for display
          type: file.type,
          size: file.size,
          isScreenshot: true,
          originalPath: file.path
        };
      }
      
      // For regular uploaded files
      if (file.file) {
        // Convert File object to buffer for Electron
        const arrayBuffer = await file.file.arrayBuffer();
        const savedPath = await window.electronAPI.saveImageBufferToTmp(arrayBuffer, uniqueFileName);
        
        // Also get base64 for display purposes (to avoid file:// security issues)
        let displayUrl = null;
        try {
          if (window.electronAPI.readImageAsBase64) {
            const base64Result = await window.electronAPI.readImageAsBase64(savedPath);
            if (base64Result.success) {
              displayUrl = base64Result.dataUrl;
            }
          }
        } catch (error) {
          console.warn('Could not read saved image as base64:', error);
        }
        
        // Fallback to base64 from original file if reading saved file fails
        if (!displayUrl) {
          try {
            displayUrl = await new Promise((resolve, reject) => {
              const reader = new FileReader();
              reader.onload = () => resolve(reader.result);
              reader.onerror = reject;
              reader.readAsDataURL(file.file);
            });
          } catch (error) {
            console.warn('Could not create base64 from original file:', error);
          }
        }
        
        return {
          name: file.name,
          path: savedPath, // File path for backend
          displayUrl: displayUrl, // Base64 URL for display
          type: file.type,
          size: file.size,
          isScreenshot: false,
          originalPath: file.path
        };
      }
      
      // For files with existing paths
      if (file.path) {
        const savedPath = await window.electronAPI.saveImageToTmp(file.path, uniqueFileName);
        
        // Also get base64 for display purposes (to avoid file:// security issues)
        let displayUrl = null;
        try {
          if (window.electronAPI.readImageAsBase64) {
            const base64Result = await window.electronAPI.readImageAsBase64(savedPath);
            if (base64Result.success) {
              displayUrl = base64Result.dataUrl;
            }
          }
        } catch (error) {
          console.warn('Could not read saved file as base64:', error);
        }
        
        return {
          name: file.name,
          path: savedPath, // File path for backend
          displayUrl: displayUrl, // Base64 URL for display
          type: file.type,
          size: file.size,
          isScreenshot: file.isScreenshot || false,
          originalPath: file.path
        };
      }
      
      return file;
    } catch (error) {
      console.error('Error saving image to local directory:', error);
      // Fallback to web handling if Electron fails
      if (file.file) {
        try {
          const base64 = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file.file);
          });
          
          return {
            name: file.name,
            path: file.path || file.name,
            displayUrl: base64, // Use base64 data URL for display
            type: file.type,
            size: file.size,
            isScreenshot: file.isScreenshot || false,
            isBase64: true,
            originalPath: file.path
          };
        } catch (fallbackError) {
          console.error('Fallback file processing also failed:', fallbackError);
        }
      }
      
      return file; // Final fallback to original file
    }
  };

  const sendMessage = async (messageText, imageFiles = []) => {
    console.log('📤 Sending message:', messageText, 'Educational mode:', isEducationalMode);
    
    if (!messageText.trim() && imageFiles.length === 0) return;

    // Save images to local directory first
    const savedImages = await Promise.all(
      imageFiles.map(file => saveImageToLocal(file))
    );

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: messageText,
      images: savedImages,
      timestamp: new Date().toISOString()
    };

    console.log('👤 Adding user message to conversation, current messages count:', messages?.length);
    console.log('🔍 User message being added:', userMessage);
    setMessages(prev => {
      console.log('📝 Previous messages:', prev?.length, prev);
      const newMessages = [...prev, userMessage];
      console.log('✅ Updated messages:', newMessages.length, newMessages);
      return newMessages;
    });
    
    // Generate unique request ID for this specific message
    const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Create abort controller for this request
    const abortController = new AbortController();
    abortControllersRef.current.set(requestId, abortController);
    
    // Add to active streaming requests
    setActiveStreamingRequests(prev => new Map([...prev, [requestId, { streamingContent: '' }]]));

    let streamEnded = false; // 移到try块外面定义
    let cleanup = null;
    try {
      let imageUris = null;
      if (savedImages.length > 0) {
        imageUris = savedImages.map(file => {
          // For base64 data (web mode), send the displayUrl which contains base64 data
          if (file.isBase64 && file.displayUrl) {
            return file.displayUrl;
          }
          // For file paths (Electron mode), send the file path for backend processing
          return file.path;
        });
      }

      // 构建请求数据，在教育模式下添加特殊的系统提示词
      const requestData = {
        message: messageText || null,
        image_uris: imageUris,
        memorizing: false
      };

      // 如果是教育模式，添加教育场景的系统提示词
      if (isEducationalMode && settings.educationalContext) {
        requestData.educational_mode = true;
        requestData.system_prompt = settings.educationalContext.systemPrompt;
        requestData.student_info = {
          name: settings.educationalContext.student.name,
          characteristics: settings.educationalContext.student.characteristics,
          weakPoints: settings.educationalContext.student.weakPoints
        };
        requestData.question_context = settings.educationalContext.question;
      }

      const result = await queuedFetch(`${settings.serverUrl}/send_streaming_message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
        signal: abortController.signal,
        isStreaming: true
      });

      const response = result.response;
      cleanup = result.cleanup;

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true && !streamEnded) {
        const { done, value } = await reader.read();
        
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'intermediate') {
                // Update streaming content for this specific request
                setActiveStreamingRequests(prev => {
                  const updated = new Map(prev);
                  const current = updated.get(requestId);
                  if (current) {
                    let newContent = current.streamingContent;
                    if (data.message_type === 'internal_monologue') {
                      newContent += '\n[Thinking] ' + data.content;
                    } else if (data.message_type === 'response') {
                      newContent += '\n' + data.content;
                    }
                    updated.set(requestId, { ...current, streamingContent: newContent });
                  }
                  return updated;
                });
              } else if (data.type === 'missing_api_keys') {
                // Handle missing API keys by showing the modal
                setMissingApiKeys(data.missing_keys);
                setCurrentModelType(data.model_type);
                setShowApiKeyModal(true);
                return; // Don't continue processing
              } else if (data.type === 'final') {
                console.log('🎯 Received final response:', data.response);
                const assistantMessage = {
                  id: Date.now() + 1,
                  type: 'assistant',
                  content: data.response,
                  timestamp: new Date().toISOString()
                };
                setMessages(prev => {
                  const newMessages = [...prev, assistantMessage];
                  console.log('💬 Final messages state:', newMessages);
                  return newMessages;
                });
                streamEnded = true; // 标记流结束，退出整个while循环
                break;
              } else if (data.type === 'error') {
                throw new Error(data.error);
              }
            } catch (parseError) {
              console.error('Error parsing SSE data:', parseError);
            }
          }
        }
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('Request was aborted');
      } else {
        console.error('Error sending message:', error);
        const errorMessage = {
          id: Date.now() + 1,
          type: 'error',
          content: `Error: ${error.message}`,
          timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } finally {
      console.log('🧹 Cleaning up request:', requestId, 'streamEnded:', streamEnded);
      
      // Clean up this request
      setActiveStreamingRequests(prev => {
        const updated = new Map(prev);
        updated.delete(requestId);
        console.log('🗑️ Removed streaming request:', requestId);
        return updated;
      });
      
      abortControllersRef.current.delete(requestId);
      
      // Call cleanup to notify request queue
      if (cleanup) {
        cleanup();
      }
    }
  };

  const clearChatLocal = () => {
    setMessages([]);
    // Abort all active requests
    abortControllersRef.current.forEach((controller) => {
      controller.abort();
    });
    abortControllersRef.current.clear();
    setActiveStreamingRequests(new Map());
    setShowClearModal(false);
  };

  const clearChatPermanent = async () => {
    setIsClearing(true);
    
    try {
      const response = await queuedFetch(`${settings.serverUrl}/conversation/clear`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to clear conversation: ${response.status}`);
      }

      const result = await response.json();
      
      // Clear local messages too
      setMessages([]);
      
      // Abort all active requests
      abortControllersRef.current.forEach((controller) => {
        controller.abort();
      });
      abortControllersRef.current.clear();
      setActiveStreamingRequests(new Map());

      // Show success message briefly
      const successMessage = {
        id: Date.now(),
        type: 'assistant',
        content: `✅ ${result.message}`,
        timestamp: new Date().toISOString()
      };
      setMessages([successMessage]);

      setShowClearModal(false);
    } catch (error) {
      console.error('Error clearing conversation:', error);
      
      // Show error message
      const errorMessage = {
        id: Date.now(),
        type: 'error',
        content: `❌ Failed to clear conversation history: ${error.message}`,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsClearing(false);
    }
  };

  const handleClearClick = () => {
    setShowClearModal(true);
  };

  const stopGeneration = () => {
    // Abort all active requests
    abortControllersRef.current.forEach((controller) => {
      controller.abort();
    });
  };

  const toggleScreenshotSetting = async () => {
    try {
      const newSetting = !includeScreenshots;
      const response = await queuedFetch(`${settings.serverUrl}/screenshot_setting/set`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          include_recent_screenshots: newSetting
        })
      });

      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setIncludeScreenshots(newSetting);
        } else {
          console.error('Error setting screenshot setting:', data.message);
        }
      } else {
        console.error('HTTP error setting screenshot setting:', response.status);
      }
    } catch (error) {
      console.error('Error toggling screenshot setting:', error);
    }
  };

  const handleApiKeySubmit = () => {
    // After API keys are submitted, the modal will close
    setShowApiKeyModal(false);
    setMissingApiKeys([]);
    setCurrentModelType('');
  };

  const closeApiKeyModal = () => {
    setShowApiKeyModal(false);
    setMissingApiKeys([]);
    setCurrentModelType('');
  };

  return (
    <div className="chat-window">
      <div className="chat-header">
        <div className="chat-info">
          <span className="model-info">Model: {currentModel}</span>
          <span className="persona-info">Persona: {settings.persona}</span>
        </div>
        <div className="chat-actions">
          <button 
            className={`screenshot-toggle ${includeScreenshots ? 'enabled' : 'disabled'}`}
            onClick={toggleScreenshotSetting}
            title={includeScreenshots ? "Allow assistant to see your recent screenshots" : "Assistant cannot see your recent screenshots"}
          >
            📷 {includeScreenshots ? 'ON' : 'OFF'}
          </button>
          {hasActiveStreaming && (
            <button 
              className="stop-button"
              onClick={stopGeneration}
              title="Stop generation"
            >
              ⏹️ Stop
            </button>
          )}
                      <button 
              className="clear-button"
              onClick={handleClearClick}
              title="Clear chat"
            >
              🗑️ Clear
            </button>
          </div>
        </div>

        <div className="messages-container">
          {messages.length === 0 && !isEducationalMode && (
            <div className="welcome-message">
              <h2>Welcome to MIRIX!</h2>
              <p>Start a conversation with your AI assistant.</p>
              {window.electronAPI ? (
                <p>💡 MIRIX is running in the desktop app environment.</p>
              ) : (
                <p>💡 Download the desktop app for an enhanced experience and more features!</p>
              )}
            </div>
          )}
          
          {console.log('🎭 Rendering messages in ChatWindow:', messages?.length, messages) || null}
          {messages.map(message => (
            <ChatBubble key={message.id} message={message} />
          ))}
          
          {currentStreamingContent && (
            <ChatBubble 
              message={{
                id: 'streaming',
                type: 'assistant',
                content: currentStreamingContent,
                timestamp: new Date().toISOString(),
                isStreaming: true
              }} 
            />
          )}
          
          <div ref={messagesEndRef} />
        </div>

        <MessageInput 
          onSendMessage={sendMessage}
          disabled={hasActiveStreaming}
        />

      <ApiKeyModal
        isOpen={showApiKeyModal}
        onClose={closeApiKeyModal}
        missingKeys={missingApiKeys}
        modelType={currentModelType}
        onSubmit={handleApiKeySubmit}
        serverUrl={settings.serverUrl}
      />

      <ClearChatModal
        isOpen={showClearModal}
        onClose={() => setShowClearModal(false)}
        onClearLocal={clearChatLocal}
        onClearPermanent={clearChatPermanent}
        isClearing={isClearing}
      />
    </div>
  );
};

export default ChatWindow; 