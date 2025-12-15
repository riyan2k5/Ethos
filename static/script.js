const API_BASE = '';
let currentSong = null;
let searchTimeout;

// Get or create user ID
function getUserId() {
    let userId = localStorage.getItem('ethos_user_id');
    if (!userId) {
        userId = 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('ethos_user_id', userId);
    }
    return userId;
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing app...');
    
    // Wait a bit to ensure everything is ready
    setTimeout(() => {
        loadHome();
    }, 100);
    
    const input = document.getElementById('search-input');
    const dropdown = document.getElementById('live-results');
    
    if (!input || !dropdown) {
        console.error('Required elements not found!');
        return;
    }
    
    input.addEventListener('input', (e) => {
        clearTimeout(searchTimeout);
        const query = e.target.value.trim();
        
        const clearBtn = document.getElementById('search-clear');
        if (clearBtn) clearBtn.style.display = query.length > 0 ? 'block' : 'none';
        
        if (query.length > 2) {
            dropdown.classList.remove('hidden');
            dropdown.innerHTML = '<div style="padding:15px;color:#888;text-align:center">Searching...</div>';
            searchTimeout = setTimeout(() => liveSearch(query), 300);
        } else {
            dropdown.classList.add('hidden');
        }
    });
    
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            dropdown.classList.add('hidden');
            runSearch();
        }
    });
    
    document.addEventListener('click', (e) => {
        if (!input.contains(e.target) && !dropdown.contains(e.target)) {
            dropdown.classList.add('hidden');
        }
    });
});

function clearSearch() {
    const input = document.getElementById('search-input');
    input.value = '';
    document.getElementById('live-results').classList.add('hidden');
    const clearBtn = document.getElementById('search-clear');
    if (clearBtn) clearBtn.style.display = 'none';
    loadHome();
}

function loadHome() {
    console.log('loadHome() called');
    const container = document.getElementById('content-area');
    if (!container) {
        console.error('Content area not found!');
        return;
    }
    
    console.log('Setting loading state...');
    container.innerHTML = '<div class="loading"><i class="fas fa-spinner"></i><p>Loading recommendations...</p></div>';
    
    console.log('Fetching recommendations and clusters...');
    Promise.all([
        fetch('/api/recommendations?limit=20'),
        fetch('/api/clusters?limit=15')
    ])
    .then(async ([recRes, clusterRes]) => {
        console.log('API responses received:', recRes.status, clusterRes.status);
        
        if (!recRes.ok) {
            throw new Error(`Recommendations API error: ${recRes.status} ${recRes.statusText}`);
        }
        if (!clusterRes.ok) {
            throw new Error(`Clusters API error: ${clusterRes.status} ${clusterRes.statusText}`);
        }
        
        const recommended = await recRes.json();
        const clusters = await clusterRes.json();
        
        console.log('Recommendations:', recommended?.length || 0);
        console.log('Clusters:', Object.keys(clusters || {}).length);
        
        container.innerHTML = '';
        
        // Recommended Songs
        if (recommended && recommended.length > 0) {
            console.log('Creating recommendations row...');
            createRow('Recommended for You', recommended, 'recommended');
        } else {
            console.warn('No recommendations returned');
        }
        
        // Clustered Songs
        if (clusters && typeof clusters === 'object') {
            for (const [clusterName, songs] of Object.entries(clusters)) {
                if (songs && Array.isArray(songs) && songs.length > 0) {
                    console.log(`Creating cluster row: ${clusterName} with ${songs.length} songs`);
                    createRow(clusterName, songs, 'cluster');
                }
            }
        } else {
            console.warn('No clusters returned or invalid format');
        }
        
        // If no content was added, show a message
        if (container.innerHTML === '') {
            console.warn('No content was added to container');
            container.innerHTML = '<div style="padding:40px;text-align:center;color:#64748b">No songs available. Please check the dataset.</div>';
        }
    })
    .catch(err => {
        console.error('Error loading home:', err);
        container.innerHTML = `<div style="padding:40px;color:#ef4444;text-align:center">
            <h2>Error loading content</h2>
            <p>${err.message}</p>
            <p style="margin-top:20px;font-size:0.9rem;color:#64748b">Please check the browser console for more details.</p>
            <button onclick="loadHome()" style="margin-top:20px;padding:10px 20px;background:#10b981;color:white;border:none;border-radius:8px;cursor:pointer;">Retry</button>
        </div>`;
    });
}

function createRow(title, items, type) {
    const container = document.getElementById('content-area');
    const section = document.createElement('section');
    section.className = 'row-wrapper';
    
    section.innerHTML = `<div class="row-title">${title}</div>`;
    const scroller = document.createElement('div');
    scroller.className = 'row-scroller';
    
    items.forEach(item => {
        const card = document.createElement('div');
        card.className = 'card';
        card.onclick = () => openSongModal(item);
        
        const trackName = item.track_name || item.title || 'Unknown';
        const artistName = item.artist_name || item.artists || 'Unknown';
        const trackId = item.track_id || item.id || '';
        
        // Use placeholder initially - load cover art lazily
        const placeholder = 'https://via.placeholder.com/200x200/10b981/ffffff?text=♪';
        
        card.innerHTML = `
            <img src="${placeholder}" class="poster" loading="lazy" alt="${trackName}" data-track-id="${trackId}">
            <div class="card-info">
                <div class="card-title">${trackName}</div>
                <div class="card-artist">${artistName}</div>
            </div>
        `;
        
        scroller.appendChild(card);
        
        // Load cover art lazily (only when card is visible or after a delay)
        if (trackId) {
            setTimeout(() => loadCoverArtForCard(card, trackId), Math.random() * 500); // Stagger requests
        }
    });
    
    section.appendChild(scroller);
    container.appendChild(section);
}

// Lazy load cover art for a single card using Intersection Observer
function loadCoverArtForCard(card, trackId) {
    const img = card.querySelector(`img[data-track-id="${trackId}"]`);
    if (!img) return;
    
    // Use Intersection Observer to load only when card is visible
    if ('IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting && entry.target.src.includes('placeholder')) {
                    observer.unobserve(entry.target);
                    // Use faster cover-only endpoint
                    fetch(`/api/cover/${trackId}`)
                        .then(res => res.json())
                        .then(data => {
                            if (data.cover_art) {
                                entry.target.src = data.cover_art;
                            }
                        })
                        .catch(() => {}); // Silently fail
                }
            });
        }, { rootMargin: '100px' }); // Start loading 100px before visible
        
        observer.observe(img);
    } else {
        // Fallback: load after random delay
        setTimeout(() => {
            fetch(`/api/cover/${trackId}`)
                .then(res => res.json())
                .then(data => {
                    if (data.cover_art && img.src.includes('placeholder')) {
                        img.src = data.cover_art;
                    }
                })
                .catch(() => {});
        }, Math.random() * 1000);
    }
}

// Removed loadCoverArts - now using lazy loading per card
// This is faster and doesn't block the UI

async function liveSearch(query) {
    const dropdown = document.getElementById('live-results');
    try {
        const res = await fetch(`/api/search?query=${encodeURIComponent(query)}&limit=8`);
        const items = await res.json();
        dropdown.innerHTML = '';
        
        if (items.length > 0) {
            items.forEach(item => {
                const div = document.createElement('div');
                div.className = 'live-item';
                const title = item.track_name || item.title || 'Unknown';
                const artist = item.artist_name || item.artists || 'Unknown';
                const trackId = item.track_id || item.id || '';
                const img = item.cover_art || 'https://via.placeholder.com/50x50/10b981/ffffff?text=♪';
                
                div.innerHTML = `
                    <img src="${img}" class="live-poster" alt="${title}">
                    <div class="live-info">
                        <span class="live-title">${title}</span>
                        <span class="live-artist">${artist}</span>
                    </div>
                `;
                div.onclick = () => {
                    dropdown.classList.add('hidden');
                    openSongModal(item);
                };
                dropdown.appendChild(div);
            });
        } else {
            dropdown.innerHTML = '<div style="padding:15px;color:#888;text-align:center">No results found.</div>';
        }
    } catch (e) {
        console.error(e);
        dropdown.innerHTML = '<div style="padding:15px;color:#ef4444;text-align:center">Search error.</div>';
    }
}

async function runSearch() {
    const query = document.getElementById('search-input').value;
    if (!query) return;
    
    const container = document.getElementById('content-area');
    container.innerHTML = '<div class="loading"><i class="fas fa-spinner"></i><p>Searching...</p></div>';
    
    try {
        const res = await fetch(`/api/search?query=${encodeURIComponent(query)}&limit=50`);
        const items = await res.json();
        container.innerHTML = '';
        
        if (items.length === 0) {
            container.innerHTML = '<div style="padding:40px;text-align:center;color:#64748b">No songs found.</div>';
            return;
        }
        
        createRow(`Search Results for "${query}"`, items, 'search');
    } catch (err) {
        container.innerHTML = '<div style="padding:40px;color:#ef4444;text-align:center">Search error. Please try again.</div>';
    }
}

async function openSongModal(song) {
    const modal = document.getElementById('song-modal');
    const trackId = song.track_id || song.id;
    
    if (!trackId) {
        alert('Song ID not found');
        return;
    }
    
    // Track interaction
    trackInteraction(trackId, 'click');
    
    modal.classList.remove('hidden');
    const modalContent = modal.querySelector('.modal-content');
    modalContent.innerHTML = '<div class="loading"><i class="fas fa-spinner"></i><p>Loading...</p></div>';
    
    try {
        // Set a timeout for the fetch (8 seconds - reduced from 10)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 8000);
        
        const res = await fetch(`/api/song/${trackId}`, {
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status} ${res.statusText}`);
        }
        
        const data = await res.json();
        
        // Validate response
        if (!data || !data.song) {
            throw new Error('Invalid response: missing song data');
        }
        
        currentSong = data.song;
        
        // Restore modal structure (it was replaced with loading)
        modalContent.innerHTML = `
            <span class="close-btn" onclick="closeModal()"><i class="fa-solid fa-xmark"></i></span>
            <div class="modal-header">
                <div class="poster-wrapper">
                    <img id="m-poster" src="${data.cover_art || 'https://via.placeholder.com/280x280/10b981/ffffff?text=Music'}" alt="Album Cover">
                </div>
                <div class="modal-info">
                    <h2 id="m-title">${data.song.track_name || data.song.title || 'Unknown'}</h2>
                    <div class="m-meta">
                        <span id="m-artist" class="pixel-badge">${data.song.artist_name || data.song.artists || 'Unknown'}</span>
                    </div>
                    <div id="ai-tags-row" class="ai-tags-row"></div>
                    <p id="m-desc">Discover this track and explore similar songs.</p>
                    <div class="control-deck">
                        <button id="play-btn" class="pixel-btn action" onclick="playOnSpotify()">
                            <i class="fab fa-spotify"></i> Play on Spotify
                        </button>
                        <button id="like-btn" class="pixel-btn" onclick="toggleLike()">
                            <i class="fa-regular fa-heart"></i>
                        </button>
                    </div>
                </div>
            </div>
            <div id="similar-section" class="similar-section">
                <h3 class="section-title">More like this</h3>
                <div id="similar-songs" class="similar-songs-grid"></div>
            </div>
        `;
        
        // AI Tags
        const tagsRow = document.getElementById('ai-tags-row');
        if (tagsRow && data.predictions && data.predictions.genre) {
            const genre = data.predictions.genre;
            if (genre.top_predictions && genre.top_predictions.length > 0) {
                genre.top_predictions.forEach(pred => {
                    const badge = document.createElement('span');
                    badge.className = 'ai-badge';
                    badge.innerHTML = `<i class="fas fa-tag"></i> ${pred.name} (${Math.round(pred.score * 100)}%)`;
                    tagsRow.appendChild(badge);
                });
            } else if (genre.predicted) {
                const badge = document.createElement('span');
                badge.className = 'ai-badge';
                badge.innerHTML = `<i class="fas fa-tag"></i> ${genre.predicted}`;
                tagsRow.appendChild(badge);
            }
        }
        
        // Similar Songs
        const similarSongs = document.getElementById('similar-songs');
        if (similarSongs) {
            // Filter out the current song itself and remove duplicates
            const currentId = data.song.track_id || data.song.id;
            const seenIds = new Set();
            const filteredSimilar = (data.similar || []).filter(sim => {
                const simId = sim.track_id || sim.id;
                // Exclude current song and duplicates
                if (simId === currentId || seenIds.has(simId)) {
                    return false;
                }
                seenIds.add(simId);
                return true;
            });
            
            if (filteredSimilar.length > 0) {
                console.log(`Displaying ${filteredSimilar.length} similar songs`);
                filteredSimilar.forEach(simSong => {
                    const card = document.createElement('div');
                    card.className = 'similar-song-card';
                    card.onclick = () => {
                        closeModal();
                        setTimeout(() => openSongModal(simSong), 300);
                    };
                    
                    // Use cover_art from API response, or placeholder
                    const imgSrc = simSong.cover_art || 'https://via.placeholder.com/150x150/10b981/ffffff?text=♪';
                    const simTitle = simSong.track_name || simSong.title || 'Unknown';
                    const simArtist = simSong.artist_name || simSong.artists || 'Unknown';
                    const simTrackId = simSong.track_id || simSong.id;
                    
                    card.innerHTML = `
                        <img src="${imgSrc}" alt="${simTitle}" data-track-id="${simTrackId}" onerror="this.src='https://via.placeholder.com/150x150/10b981/ffffff?text=♪'">
                        <div class="similar-song-info">
                            <div class="similar-song-title">${simTitle}</div>
                            <div class="similar-song-artist">${simArtist}</div>
                        </div>
                    `;
                    similarSongs.appendChild(card);
                    
                    // Load cover art lazily using Intersection Observer (only when visible)
                    if (simTrackId) {
                        const img = card.querySelector(`img[data-track-id="${simTrackId}"]`);
                        if (img && 'IntersectionObserver' in window) {
                            const observer = new IntersectionObserver((entries) => {
                                entries.forEach(entry => {
                                    if (entry.isIntersecting && entry.target.src.includes('placeholder')) {
                                        observer.unobserve(entry.target);
                                        // Use faster cover-only endpoint
                                        fetch(`/api/cover/${simTrackId}`)
                                            .then(res => res.json())
                                            .then(songData => {
                                                if (songData.cover_art) {
                                                    entry.target.src = songData.cover_art;
                                                }
                                            })
                                            .catch(() => {}); // Silently fail
                                    }
                                });
                            }, { rootMargin: '50px' });
                            observer.observe(img);
                        } else {
                            // Fallback: load after delay
                            setTimeout(() => {
                                fetch(`/api/cover/${simTrackId}`)
                                    .then(res => res.json())
                                    .then(songData => {
                                        if (songData.cover_art && img.src.includes('placeholder')) {
                                            img.src = songData.cover_art;
                                        }
                                    })
                                    .catch(() => {});
                            }, 500);
                        }
                    }
                });
            } else {
                console.log('No similar songs found');
                similarSongs.innerHTML = '<div style="padding:20px;text-align:center;color:#64748b">No similar songs found.</div>';
            }
        } else {
            console.error('similar-songs element not found!');
        }
        
    } catch (err) {
        console.error('Error loading song details:', err);
        let errorMsg = 'Error loading song details.';
        if (err.name === 'AbortError') {
            errorMsg = 'Request timed out. The server may be slow. Please try again.';
        } else if (err.message) {
            errorMsg = `Error: ${err.message}`;
        }
        modalContent.innerHTML = `<div style="padding:40px;color:#ef4444;text-align:center">
            <h2>Error Loading Song</h2>
            <p>${errorMsg}</p>
            <button onclick="closeModal(); setTimeout(() => openSongModal(${JSON.stringify(song).replace(/"/g, '&quot;')}), 100)" 
                    style="margin-top:20px;padding:10px 20px;background:#10b981;color:white;border:none;border-radius:8px;cursor:pointer;">
                Retry
            </button>
        </div>`;
    }
}

function closeModal() {
    document.getElementById('song-modal').classList.add('hidden');
    currentSong = null;
}

function backdropClose(e) {
    if (e.target.id === 'song-modal') {
        closeModal();
    } else if (e.target.id === 'settings-modal') {
        closeSettings();
    }
}

function openSettings() {
    const modal = document.getElementById('settings-modal');
    modal.classList.remove('hidden');
    loadAnalytics();
}

function closeSettings() {
    const modal = document.getElementById('settings-modal');
    modal.classList.add('hidden');
}

async function loadAnalytics() {
    const content = document.getElementById('analytics-content');
    content.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i><p>Loading analytics...</p></div>';
    
    try {
        const res = await fetch('/api/analytics');
        const data = await res.json();
        
        let html = '';
        
        // Dataset info
        if (data.dataset_info) {
            html += `
                <div class="dataset-info">
                    <h3><i class="fas fa-database"></i> Current Dataset</h3>
                    <div class="dataset-stats">
                        <div class="dataset-stat">
                            <div class="dataset-stat-label">Songs Available</div>
                            <div class="dataset-stat-value">${data.dataset_info.current_dataset_size || 0}</div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Model cards
        if (data.models && data.models.length > 0) {
            data.models.forEach(model => {
                const statusClass = model.exists ? 'active' : 'inactive';
                const statusText = model.exists ? 'Active' : 'Not Found';
                const statusIcon = model.exists ? 'fa-check-circle' : 'fa-times-circle';
                
                html += `
                    <div class="model-card">
                        <div class="model-header">
                            <div>
                                <div class="model-name">${model.name}</div>
                                <div class="model-type">${model.type}</div>
                                <div class="model-algorithm"><i class="fas fa-brain"></i> ${model.algorithm}</div>
                            </div>
                            <span class="model-status ${statusClass}">
                                <i class="fas ${statusIcon}"></i> ${statusText}
                            </span>
                        </div>
                        ${model.exists ? `
                            <div class="model-details">
                                <div class="detail-item">
                                    <div class="detail-label">Model Size</div>
                                    <div class="detail-value">${model.model_file || 'N/A'}</div>
                                </div>
                                ${model.scaler_file ? `
                                    <div class="detail-item">
                                        <div class="detail-label">Scaler Size</div>
                                        <div class="detail-value">${model.scaler_file}</div>
                                    </div>
                                ` : ''}
                                <div class="detail-item">
                                    <div class="detail-label">Features</div>
                                    <div class="detail-value">${model.num_features || 'N/A'}</div>
                                </div>
                                <div class="detail-item">
                                    <div class="detail-label">Last Trained</div>
                                    <div class="detail-value" style="font-size: 0.875rem;">${model.model_date || 'N/A'}</div>
                                </div>
                                ${model.n_estimators ? `
                                    <div class="detail-item">
                                        <div class="detail-label">Estimators</div>
                                        <div class="detail-value">${model.n_estimators}</div>
                                    </div>
                                ` : ''}
                                ${model.n_clusters ? `
                                    <div class="detail-item">
                                        <div class="detail-label">Clusters</div>
                                        <div class="detail-value">${model.n_clusters}</div>
                                    </div>
                                ` : ''}
                                ${model.n_neighbors ? `
                                    <div class="detail-item">
                                        <div class="detail-label">Neighbors</div>
                                        <div class="detail-value">${model.n_neighbors}</div>
                                    </div>
                                ` : ''}
                                ${model.num_classes ? `
                                    <div class="detail-item">
                                        <div class="detail-label">Classes</div>
                                        <div class="detail-value">${model.num_classes}</div>
                                    </div>
                                ` : ''}
                                ${model.num_tracks ? `
                                    <div class="detail-item">
                                        <div class="detail-label">Tracks in Model</div>
                                        <div class="detail-value">${model.num_tracks.toLocaleString()}</div>
                                    </div>
                                ` : ''}
                            </div>
                            ${model.metrics && Object.keys(model.metrics).length > 0 ? `
                                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 2px solid var(--primary-green);">
                                    <div class="detail-label" style="font-size: 0.875rem; margin-bottom: 0.75rem; color: var(--primary-green); font-weight: 700;">
                                        <i class="fas fa-chart-bar"></i> Evaluation Metrics
                                    </div>
                                    <div class="model-details">
                                        ${model.metrics.accuracy !== null && model.metrics.accuracy !== undefined ? `
                                            <div class="detail-item" style="background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%);">
                                                <div class="detail-label">Accuracy</div>
                                                <div class="detail-value" style="color: var(--primary-green); font-size: 1.25rem;">
                                                    ${(model.metrics.accuracy * 100).toFixed(2)}%
                                                </div>
                                            </div>
                                        ` : ''}
                                        ${model.metrics.r2_score !== null && model.metrics.r2_score !== undefined ? `
                                            <div class="detail-item" style="background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%);">
                                                <div class="detail-label">R² Score</div>
                                                <div class="detail-value" style="color: var(--primary-green); font-size: 1.25rem;">
                                                    ${model.metrics.r2_score.toFixed(4)}
                                                </div>
                                            </div>
                                        ` : ''}
                                        ${model.metrics.mse !== null && model.metrics.mse !== undefined ? `
                                            <div class="detail-item">
                                                <div class="detail-label">MSE</div>
                                                <div class="detail-value">${model.metrics.mse.toFixed(4)}</div>
                                            </div>
                                        ` : ''}
                                        ${model.metrics.rmse !== null && model.metrics.rmse !== undefined ? `
                                            <div class="detail-item">
                                                <div class="detail-label">RMSE</div>
                                                <div class="detail-value">${model.metrics.rmse.toFixed(4)}</div>
                                            </div>
                                        ` : ''}
                                        ${model.metrics.mae !== null && model.metrics.mae !== undefined ? `
                                            <div class="detail-item">
                                                <div class="detail-label">MAE</div>
                                                <div class="detail-value">${model.metrics.mae.toFixed(4)}</div>
                                            </div>
                                        ` : ''}
                                        ${model.metrics.silhouette_score !== null && model.metrics.silhouette_score !== undefined ? `
                                            <div class="detail-item" style="background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%);">
                                                <div class="detail-label">Silhouette Score</div>
                                                <div class="detail-value" style="color: var(--primary-green); font-size: 1.25rem;">
                                                    ${model.metrics.silhouette_score.toFixed(4)}
                                                </div>
                                            </div>
                                        ` : ''}
                                        ${model.metrics.precision !== null && model.metrics.precision !== undefined ? `
                                            <div class="detail-item">
                                                <div class="detail-label">Precision</div>
                                                <div class="detail-value">${(model.metrics.precision * 100).toFixed(2)}%</div>
                                            </div>
                                        ` : ''}
                                        ${model.metrics.recall !== null && model.metrics.recall !== undefined ? `
                                            <div class="detail-item">
                                                <div class="detail-label">Recall</div>
                                                <div class="detail-value">${(model.metrics.recall * 100).toFixed(2)}%</div>
                                            </div>
                                        ` : ''}
                                        ${model.metrics.f1_score !== null && model.metrics.f1_score !== undefined ? `
                                            <div class="detail-item">
                                                <div class="detail-label">F1 Score</div>
                                                <div class="detail-value">${(model.metrics.f1_score * 100).toFixed(2)}%</div>
                                            </div>
                                        ` : ''}
                                    </div>
                                    ${model.metrics.note ? `
                                        <div style="margin-top: 0.75rem; padding: 0.75rem; background: #fef3c7; border-radius: 8px; font-size: 0.875rem; color: #92400e;">
                                            <i class="fas fa-info-circle"></i> ${model.metrics.note}
                                        </div>
                                    ` : ''}
                                </div>
                            ` : ''}
                            ${model.classes && model.classes.length > 0 ? `
                                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;">
                                    <div class="detail-label">Sample Genres:</div>
                                    <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem;">
                                        ${model.classes.map(c => `<span class="pixel-badge">${c}</span>`).join('')}
                                    </div>
                                </div>
                            ` : ''}
                        ` : `
                            <div style="padding: 1rem; text-align: center; color: var(--text-secondary);">
                                Model file not found. Please train the model first.
                            </div>
                        `}
                    </div>
                `;
            });
        } else {
            html = '<div style="padding: 2rem; text-align: center; color: var(--text-secondary);">No model data available.</div>';
        }
        
        content.innerHTML = html;
    } catch (err) {
        console.error('Error loading analytics:', err);
        content.innerHTML = '<div style="padding: 2rem; text-align: center; color: #ef4444;">Error loading analytics. Please try again.</div>';
    }
}

function playOnSpotify() {
    if (currentSong) {
        const trackId = currentSong.track_id || currentSong.id;
        if (trackId) {
            window.open(`https://open.spotify.com/track/${trackId}`, '_blank');
        }
    }
}

function toggleLike() {
    if (!currentSong) return;
    
    const btn = document.getElementById('like-btn');
    const isLiked = btn.classList.contains('liked');
    
    if (isLiked) {
        btn.classList.remove('liked');
        btn.innerHTML = '<i class="fa-regular fa-heart"></i>';
    } else {
        btn.classList.add('liked');
        btn.innerHTML = '<i class="fa-solid fa-heart"></i>';
    }
    
    const trackId = currentSong.track_id || currentSong.id;
    trackInteraction(trackId, isLiked ? 'unlike' : 'like');
}

function trackInteraction(trackId, action) {
    fetch('/api/interact', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            track_id: trackId,
            action: action
        })
    }).catch(err => console.error('Error tracking interaction:', err));
}

