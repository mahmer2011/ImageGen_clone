/**
 * Spine Animation Viewer
 * 
 * Uses PixiJS and pixi-spine to render and control Spine animations in the browser.
 */

class SpineViewer {
    constructor(canvasElementId, width = 800, height = 600) {
        this.app = null;
        this.spine = null;
        this.canvasElementId = canvasElementId;
        this.width = width;
        this.height = height;
        this.currentAnimation = null;
        this.isLoaded = false;
    }

    /**
     * Initialize the PixiJS application
     */
    async init() {
        const canvas = document.getElementById(this.canvasElementId);
        if (!canvas) {
            console.error(`Canvas element ${this.canvasElementId} not found`);
            return false;
        }

        // Clean up any previous instance
        if (this.app) {
            this.app.destroy(true, { children: true, texture: true, baseTexture: true });
            this.app = null;
        }

        try {
            // Check if PIXI and pixi-spine are loaded
            if (typeof PIXI === 'undefined') {
                throw new Error('PIXI is not loaded. Check if pixi.js script is included.');
            }
            
            if (!PIXI.spine) {
                console.warn('PIXI.spine not found. Checking for alternative...');
                if (window.PIXI && window.PIXI.spine) {
                    console.log('Found PIXI.spine on window object');
                } else {
                    throw new Error('PIXI.spine is not available. Check if pixi-spine script is included.');
                }
            }

            // PixiJS v6 uses synchronous constructor with options (no async init)
            this.app = new PIXI.Application({
                view: canvas,
                width: this.width,
                height: this.height,
                backgroundColor: 0xf0f0f0,
                antialias: true,
                preserveDrawingBuffer: true
            });

            console.log('Spine viewer initialized');
            console.log('PIXI version:', PIXI.VERSION);
            console.log('PIXI.spine available:', !!PIXI.spine);
            return true;
        } catch (error) {
            console.error('Failed to initialize Spine viewer:', error);
            return false;
        }
    }

    /**
     * Load a Spine character from JSON and atlas
     */
    async loadSpineCharacter(jsonPath, atlasPath) {
        if (!this.app) {
            console.error('Viewer not initialized');
            return false;
        }

        try {
            console.log(`Loading Spine data from ${jsonPath} and ${atlasPath}`);

            // Verify files exist and are accessible
            try {
                console.log('Verifying files exist...');
                const jsonResponse = await fetch(jsonPath, { method: 'HEAD' });
                console.log('JSON file check:', jsonResponse.status, jsonResponse.statusText);
                if (!jsonResponse.ok) {
                    throw new Error(`JSON file not found: ${jsonPath} (${jsonResponse.status})`);
                }
                const atlasResponse = await fetch(atlasPath, { method: 'HEAD' });
                console.log('Atlas file check:', atlasResponse.status, atlasResponse.statusText);
                if (!atlasResponse.ok) {
                    throw new Error(`Atlas file not found: ${atlasPath} (${atlasResponse.status})`);
                }
                console.log('Files verified, loading Spine data...');
            } catch (fetchError) {
                console.error('File verification error:', fetchError);
                throw new Error(`Cannot access Spine files: ${fetchError.message}`);
            }

            const spineData = await this._loadSpineData(jsonPath, atlasPath);

            if (!spineData) {
                throw new Error('Spine data was not returned by loader');
            }

            console.log('Spine data loaded, creating Spine object...');

            // Create Spine object from loaded resource
            // Try different ways to create Spine object
            let spine = null;
            if (PIXI.spine && PIXI.spine.Spine) {
                spine = new PIXI.spine.Spine(spineData);
            } else if (PIXI.spine && typeof PIXI.spine === 'function') {
                spine = new PIXI.spine(spineData);
            } else {
                throw new Error('PIXI.spine.Spine is not available. Check if pixi-spine is loaded correctly.');
            }

            if (!spine) {
                throw new Error('Failed to create Spine object');
            }

            this.spine = spine;
            console.log('Spine object created');

            // Center the character
            this.spine.x = this.width / 2;
            this.spine.y = this.height / 2;

            // Scale to fit
            const bounds = this.spine.getBounds();
            console.log('Spine bounds:', bounds);
            if (bounds && bounds.width > 0 && bounds.height > 0) {
                const scale = Math.min(
                    this.width / bounds.width,
                    this.height / bounds.height
                ) * 0.8; // Use 80% of available space
                this.spine.scale.set(scale);
                console.log('Spine scaled to:', scale);
            } else {
                // Default scale if bounds are invalid
                this.spine.scale.set(1.0);
                console.warn('Invalid bounds, using default scale');
            }

            // Add to stage
            this.app.stage.addChild(this.spine);

            // Set default state
            if (this.spine.state) {
                this.spine.state.timeScale = 1.0;
            }

            this.isLoaded = true;
            console.log('Spine character loaded successfully');
            console.log('Available animations:', this.getAvailableAnimations());

            return true;
        } catch (error) {
            console.error('Failed to load Spine character:', error);
            console.error('Error details:', error?.message, error?.stack);
            throw error; // Re-throw so caller can handle it
        }
    }

    /**
     * Internal helper to load Spine JSON + atlas via PIXI Loader
     * Updated for PIXI v6 compatibility
     */
    _loadSpineData(jsonPath, atlasPath) {
        return new Promise((resolve, reject) => {
            // Add timeout
            const timeout = setTimeout(() => {
                reject(new Error('Spine loading timeout after 30 seconds. Check browser console for details.'));
            }, 30000);

            console.log('Loading Spine data:', { jsonPath, atlasPath });
            console.log('Available PIXI APIs:', {
                hasAssets: !!PIXI.Assets,
                hasLoader: !!PIXI.Loader,
                hasLoaders: !!(PIXI.loaders && PIXI.loaders.Loader),
                hasSpine: !!PIXI.spine
            });

            // Try legacy Loader API first (most compatible with pixi-spine UMD)
            const LoaderClass = PIXI.Loader || (PIXI.loaders && PIXI.loaders.Loader);
            if (LoaderClass) {
                console.log('Using PIXI.Loader API');
                const loader = new LoaderClass();
                const spineKey = `spineData-${Date.now()}`;

                // Add progress tracking
                loader.onProgress.add((loader) => {
                    console.log(`Loading progress: ${Math.round(loader.progress)}%`);
                });

                // Add aggressive timeout check (5 seconds)
                let progressTimeout = setTimeout(() => {
                    if (loader.loading) {
                        console.error('Loader appears stuck after 5 seconds, forcing timeout');
                        console.error('Loader state:', {
                            loading: loader.loading,
                            progress: loader.progress,
                            resources: Object.keys(loader.resources || {})
                        });
                        loader.destroy();
                        clearTimeout(timeout);
                        reject(new Error('Loader timeout after 5 seconds. The pixi-spine loader may not be registered. Check browser console for PIXI.spine availability.'));
                    }
                }, 5000);

                loader.add(spineKey, jsonPath, {
                    metadata: {
                        spineAtlasFile: atlasPath
                    }
                });

                console.log('Starting loader with key:', spineKey, 'jsonPath:', jsonPath, 'atlasPath:', atlasPath);

                loader.load((loader, resources) => {
                    clearTimeout(timeout);
                    clearTimeout(progressTimeout);
                    try {
                        console.log('Loader completed. Resources:', Object.keys(resources));
                        const resource = resources[spineKey];
                        console.log('Spine resource:', resource);
                        console.log('Resource keys:', resource ? Object.keys(resource) : 'null');
                        
                        if (!resource) {
                            reject(new Error(`Resource '${spineKey}' not found in loaded resources. Available: ${Object.keys(resources).join(', ')}`));
                            return;
                        }
                        
                        if (resource.spineData) {
                            console.log('Found spineData in resource');
                            resolve(resource.spineData);
                        } else if (resource.data) {
                            console.log('Using resource.data');
                            resolve(resource.data);
                        } else {
                            console.error('Resource structure:', resource);
                            console.error('Resource type:', typeof resource);
                            console.error('Resource constructor:', resource?.constructor?.name);
                            reject(new Error('Spine resource missing spineData. Resource keys: ' + Object.keys(resource).join(', ')));
                        }
                    } catch (err) {
                        console.error('Error processing loaded resource:', err);
                        reject(err);
                    } finally {
                        loader.destroy();
                    }
                });

                loader.onError.add((error, loader, resource) => {
                    clearTimeout(timeout);
                    clearTimeout(progressTimeout);
                    console.error('Loader error:', error);
                    console.error('Error type:', typeof error);
                    console.error('Error message:', error?.message);
                    console.error('Failed resource:', resource);
                    loader.destroy();
                    const errorMsg = error?.message || error?.toString() || 'Unknown loader error';
                    reject(new Error(`Spine loader error: ${errorMsg}`));
                });
                
                return;
            }

            // Fallback: Try Assets API
            if (PIXI.Assets && PIXI.Assets.load) {
                console.log('Using PIXI.Assets API (fallback)');
                PIXI.Assets.load({
                    alias: 'spineData',
                    src: jsonPath,
                    data: {
                        spineAtlasFile: atlasPath
                    }
                }).then((resource) => {
                    clearTimeout(timeout);
                    console.log('Assets resource loaded:', resource);
                    if (resource && resource.spineData) {
                        resolve(resource.spineData);
                    } else if (resource && resource.spine) {
                        resolve(resource.spine);
                    } else {
                        reject(new Error('Spine resource loaded but missing spineData'));
                    }
                }).catch((error) => {
                    clearTimeout(timeout);
                    console.error('PIXI.Assets load error:', error);
                    reject(error);
                });
                return;
            }

            clearTimeout(timeout);
            reject(new Error('No PIXI loader available. Check if pixi.js and pixi-spine are loaded correctly.'));
        });
    }

    /**
     * Get list of available animations
     */
    getAvailableAnimations() {
        if (!this.spine || !this.spine.skeleton || !this.spine.skeleton.data) {
            return [];
        }
        
        const animations = this.spine.skeleton.data.animations;
        return animations.map(anim => anim.name);
    }

    /**
     * Play an animation by name
     */
    playAnimation(animationName, loop = true, trackIndex = 0) {
        if (!this.spine || !this.spine.state) {
            console.error('Spine not loaded');
            return false;
        }

        try {
            // Check if animation exists
            const animations = this.getAvailableAnimations();
            if (!animations.includes(animationName)) {
                console.error(`Animation '${animationName}' not found. Available: ${animations.join(', ')}`);
                return false;
            }

            // Set animation
            this.spine.state.setAnimation(trackIndex, animationName, loop);
            this.currentAnimation = animationName;

            console.log(`Playing animation: ${animationName} (loop: ${loop})`);
            return true;
        } catch (error) {
            console.error(`Failed to play animation '${animationName}':`, error);
            return false;
        }
    }

    /**
     * Stop current animation
     */
    stopAnimation() {
        if (this.spine && this.spine.state) {
            this.spine.state.clearTracks();
            this.currentAnimation = null;
        }
    }

    /**
     * Set animation playback speed
     */
    setSpeed(speed) {
        if (this.spine && this.spine.state) {
            this.spine.state.timeScale = speed;
        }
    }

    /**
     * Get current animation name
     */
    getCurrentAnimation() {
        return this.currentAnimation;
    }

    /**
     * Destroy the viewer and clean up resources
     */
    destroy() {
        if (this.spine) {
            this.spine.destroy();
            this.spine = null;
        }
        if (this.app) {
            this.app.destroy(true, { children: true, texture: true, baseTexture: true });
            this.app = null;
        }
        this.isLoaded = false;
    }

    /**
     * Set character position
     */
    setPosition(x, y) {
        if (this.spine) {
            this.spine.x = x;
            this.spine.y = y;
        }
    }

    /**
     * Set character scale
     */
    setScale(scale) {
        if (this.spine) {
            this.spine.scale.set(scale);
        }
    }
}

// Make SpineViewer available globally
window.SpineViewer = SpineViewer;

