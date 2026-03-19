// Service Worker pour Dakar Immo AI
self.addEventListener('install', (event) => {
    console.log('PWA Service Worker : Installé');
});

self.addEventListener('fetch', (event) => {
    // On laisse passer les requêtes normalement pour l'instant
    event.respondWith(fetch(event.request));
});