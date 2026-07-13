import { useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { useCreatePluginViewModel } from 'molstar/lib/extensions/plugin/hooks/use-view-model';
import { PluginCanvas } from 'molstar/lib/extensions/plugin/react';
import { loadPdb } from 'molstar/lib/extensions/plugin/loaders';
import { connectLive } from './live';

const DEFAULT_WS = 'ws://127.0.0.1:8787';

function App() {
    const model = useCreatePluginViewModel();

    useEffect(() => {
        const params = new URLSearchParams(window.location.search);
        // `?ws=ws://host:port` streams live coordinates from a Python LiveSession.
        // With no `ws` param we fall back to a static PDB so the page is never blank.
        const wsParam = params.get('ws');
        if (wsParam !== null) {
            const url = wsParam === '' ? DEFAULT_WS : wsParam;
            const handle = connectLive(model.plugin, url);
            return () => handle.close();
        }
        loadPdb(model.plugin, '1tqn');
    }, [model]);

    return (
        <div style={{ position: 'absolute', inset: 0 }}>
            <PluginCanvas model={model} />
        </div>
    );
}

createRoot(document.getElementById('app')!).render(<App />);
