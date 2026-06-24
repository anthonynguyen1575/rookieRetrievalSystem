/**
 * RAG Playground JavaScript
 *
 * Adapted for the two-panel layout.
 */

// Default system prompt
const DEFAULT_SYSTEM_PROMPT = `You are a helpful AI assistant that answers questions based on the provided context documents.

Key guidelines:
- Answer directly and concisely
- Cite specific documents when possible
- If the context doesn't contain the answer, say so
- Be honest about uncertainty`;

// Empty variables to hold current question and retrieved context (for preview purposes)
let currentQuestion = '';
let currentContextDocs = [];

/**
 * Initialize the playground with saved settings
 */
function initializePlayground() {
    // Load saved system prompt or use default
    const savedPrompt = localStorage.getItem('systemPrompt');
    document.getElementById('systemPrompt').value = savedPrompt || DEFAULT_SYSTEM_PROMPT;

    // Setup slider value displays
    updateSliderValues();

    // Add slider listeners
    document.getElementById('contextDocs').addEventListener('input', updateSliderValues);
    document.getElementById('temperature').addEventListener('input', updateSliderValues);

    // Save system prompt on change
    document.getElementById('systemPrompt').addEventListener('input', saveSystemPrompt);

    // Allow Enter key in question input
    document.getElementById('questionInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            generateAnswer();
        }
    });
}

/**
 * Update slider value displays
 */
function updateSliderValues() {
    const contextDocs = document.getElementById('contextDocs').value;
    const temperature = document.getElementById('temperature').value;

    document.getElementById('contextDocsValue').textContent = contextDocs;
    document.getElementById('temperatureValue').textContent = temperature;
}

/**
 * Save system prompt to localStorage
 */
function saveSystemPrompt() {
    const prompt = document.getElementById('systemPrompt').value;
    localStorage.setItem('systemPrompt', prompt);
}

/**
 * Reset system prompt to default
 */
function resetSystemPrompt() {
    if (confirm('Reset system prompt to default?')) {
        document.getElementById('systemPrompt').value = DEFAULT_SYSTEM_PROMPT;
        saveSystemPrompt();
    }
}

/**
 * Show prompt preview modal
 */
function showPromptPreview() {
    const question = document.getElementById('questionInput').value.trim() || '[Your question here]';
    const systemPrompt = document.getElementById('systemPrompt').value;

    const preview = `SYSTEM PROMPT:
${systemPrompt}

USER PROMPT:
Context information from relevant documents:

[Retrieved context documents will appear here]

Based only on the context above, answer the question below.
If the context does not contain sufficient information, say so clearly.

Question: ${question}

Answer:`;

    document.getElementById('promptPreviewContent').textContent = preview;
    document.getElementById('promptModal').style.display = 'block';
}

/**
 * Close prompt preview modal
 */
function closePromptModal() {
    document.getElementById('promptModal').style.display = 'none';
}

/**
 * Show complete prompt modal with system prompt, context, and question
 */
function showFullPrompt() {
    if (!currentQuestion) {
        alert('Please generate an answer first to see the complete prompt');
        return;
    }

    const systemPrompt = document.getElementById('systemPrompt').value;
    const context = formatContextDocs(currentContextDocs);
    
    const fullPrompt = `
=== SYSTEM PROMPT ===
${systemPrompt}

=== USER PROMPT ===
Context information from relevant documents:

${context}

Based only on the context above, answer the question below.
If the context does not contain sufficient information, say so clearly.

Question: ${currentQuestion}

Answer:`;

    document.getElementById('fullPromptContent').textContent = fullPrompt;
    document.getElementById('fullPromptModal').style.display = 'block';
}

/**
 * Close full prompt modal
 */
function closeFullPromptModal() {
    document.getElementById('fullPromptModal').style.display = 'none';
}

/**
 * Format context documents as they appear in the prompt
 */
function formatContextDocs(contextDocs) {
    if (!contextDocs || contextDocs.length === 0) {
        return '[No context documents retrieved]';
    }

    return contextDocs.map((doc, idx) => {
        const source = doc.source || 'Unknown';
        const text = doc.text || 'No text available';
        return `[Document ${idx + 1}] (Source: ${source})\n${text}`;
    }).join('\n\n');
}

/**
 * Toggle context card expansion
 */
function toggleContext(button) {
    const card = button.closest('.context-card');

    if (card.classList.contains('expanded')) {
        card.classList.remove('expanded');
        button.textContent = 'Show More';
    } else {
        card.classList.add('expanded');
        button.textContent = 'Show Less';
    }
}

/**
 * Generate answer using RAG
 */
async function generateAnswer() {
    const question = document.getElementById('questionInput').value.trim();
    const generateButton = document.getElementById('generateButton');
    const responseDisplay = document.getElementById('responseDisplay');
    const emptyState = document.getElementById('emptyState');
    const systemPrompt = document.getElementById('systemPrompt').value;

    // Validate question
    if (!question) {
        alert('Please enter a question');
        return;
    }

    // Get configuration
    const contextDocs = parseInt(document.getElementById('contextDocs').value);
    const temperature = parseFloat(document.getElementById('temperature').value);
    const llmProvider = document.getElementById('llmProvider').value;

    // Show loading state
    generateButton.disabled = true;
    generateButton.textContent = 'Generating...';
    responseDisplay.style.display = 'none';
    emptyState.innerHTML = '<p class="loading">Generating answer...</p>';
    emptyState.style.display = 'block';

    const startTime = performance.now();

    try {
        const response = await fetch('http://localhost:8000/rag', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                n_context_docs: contextDocs,
                temperature: temperature,
                system_prompt: systemPrompt,
                llm_provider: llmProvider
            })
        });

        const endTime = performance.now();
        const duration = ((endTime - startTime) / 1000).toFixed(2);

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Failed to generate answer');
        }

        // Display results
        displayResponse(data, duration);

    } catch (error) {
        emptyState.innerHTML = `<p class="error">Error: ${error.message}</p>`;
    } finally {
        generateButton.disabled = false;
        generateButton.textContent = 'Generate Answer';
    }
}

/**
 * Display the RAG response
 */
function displayResponse(data, duration) {
    const responseDisplay = document.getElementById('responseDisplay');
    const emptyState = document.getElementById('emptyState');
    const answerText = document.getElementById('answerText');
    const contextDocuments = document.getElementById('contextDocuments');
    const responseMetadata = document.getElementById('responseMetadata');

    // Hide empty state, show response
    emptyState.style.display = 'none';
    responseDisplay.style.display = 'block';

    // Store for full prompt preview
    currentQuestion = data.question;
    currentContextDocs = data.context || [];

    // Display answer
    answerText.textContent = data.answer;

    // Display metadata
    responseMetadata.innerHTML = `
        <div class="metadata-item">
            <span>⏱️ ${duration}s</span>
        </div>
        <div class="metadata-item">
            <span>📚 ${data.context_count} context(s)</span>
        </div>
        <div class="metadata-item">
            <span>📝 ${data.answer.split(' ').length} words</span>
        </div>
        <div class="metadata-item">
            <button onclick="showFullPrompt()" class="btn-small" style="padding: 0.25rem 0.75rem;">
                View Full Prompt
            </button>
        </div>
    `;

    // Display context documents
    if (data.context && data.context.length > 0) {
        contextDocuments.innerHTML = data.context.map((ctx, idx) => {
            const metrics = [];
            
            if (ctx.distance !== undefined && ctx.distance !== null) {
                metrics.push(`Distance: ${ctx.distance.toFixed(4)}`);
            }
            if (ctx.bm25_score !== undefined && ctx.bm25_score !== null) {
                metrics.push(`BM25: ${ctx.bm25_score.toFixed(4)}`);
            }
            if (ctx.final_score !== undefined && ctx.final_score !== null) {
                metrics.push(`Final: ${ctx.final_score.toFixed(4)}`);
            }
            if (ctx.rerank_score !== undefined && ctx.rerank_score !== null) {
                metrics.push(`Rerank: ${ctx.rerank_score.toFixed(4)}`);
            }

            let primaryMetric = 'N/A';
            
            if (ctx.rerank_score !== undefined && ctx.rerank_score !== null) {
                primaryMetric = `Rerank: ${ctx.rerank_score.toFixed(3)}`;
            } else if (ctx.distance !== undefined && ctx.distance !== null) {
                primaryMetric = `Distance: ${ctx.distance.toFixed(3)}`;
            } else if (ctx.bm25_score !== undefined && ctx.bm25_score !== null) {
                primaryMetric = `BM25: ${ctx.bm25_score.toFixed(3)}`;
            }

            const fullText = escapeHtml(ctx.text || 'No text available');
            const preview = fullText.split('\n')[0].slice(0, 120);

            return `
            <div class="context-card">
                <div class="context-header">
                    <span class="context-source">${escapeHtml(ctx.source || 'Unknown')}</span>
                    <span class="context-score">${primaryMetric}</span>
                </div>

                <div class="context-preview">
                    ${preview}${fullText.length > preview.length ? '…' : ''}
                </div>

                <button class="expand-btn" onclick="toggleContext(this)">
                    Show More
                </button>

                <div class="context-full">
                    ${fullText}
                </div>

                <div class="metadata-row">
                    ${ctx.page !== undefined && ctx.page !== null ? `<span class="metadata-item">Page ${ctx.page}</span>` : ''}
                    ${ctx.chunk_index !== undefined && ctx.chunk_index !== null ? `<span class="metadata-item">Chunk ${ctx.chunk_index}</span>` : ''}
                    ${metrics.length > 0 ? `<span class="metadata-item">${metrics.join(' | ')}</span>` : ''}
                </div>
            </div>
            `;
        }).join('');
    } else {
        contextDocuments.innerHTML = '<p class="empty-state">No context documents available</p>';
    }
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Check server health
 */
async function displayHealth() {
    try {
        const result = document.getElementById('health-status');
        result.innerHTML = '<p>Checking...</p>';

        const response = await fetch('http://localhost:8000/health?component=rag');
        const data = await response.json();

        result.innerHTML = `
            <p><strong>${data.status}</strong> — ${data.documents_indexed} chunks indexed</p>
            <p>${data.message}</p>
        `;
    } catch (error) {
        document.getElementById('health-status').textContent = 'Error: Cannot connect to server';
    }
}

// Close modals when clicking outside
window.onclick = (event) => {
    const promptModal = document.getElementById('promptModal');
    const fullPromptModal = document.getElementById('fullPromptModal');
    
    if (event.target === promptModal) {
        closePromptModal();
    }
    if (event.target === fullPromptModal) {
        closeFullPromptModal();
    }
};

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
    initializePlayground();
});