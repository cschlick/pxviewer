import { useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { MolViewSpecBehavior } from 'molstar/lib/extensions/mvs/behavior';
import { useCreatePluginViewModel } from 'molstar/lib/extensions/plugin/hooks/use-view-model';
import { PluginCanvas } from 'molstar/lib/extensions/plugin/react';
import { loadMVSFromUrl, loadPdb } from 'molstar/lib/extensions/plugin/loaders';
import { connectLive } from './live';

const DEFAULT_WS = 'ws://127.0.0.1:8787';

function App() {
    const model = useCreatePluginViewModel({
        spec: (s) => {
            s.behaviors.push(MolViewSpecBehavior);
            return s;
        },
    });

    useEffect(() => {
        const params = new URLSearchParams(window.location.search);
        // `?ws=ws://host:port` streams live coordinates from a Python LiveSession.
        // `?mvsj=path/to/scene.mvsj` loads a static MVSJ scene (e.g. a volume demo).
        // With no `ws` or `mvsj` param we fall back to a static PDB so the page is never blank.
        const wsParam = params.get('ws');
        if (wsParam !== null) {
            const url = wsParam === '' ? DEFAULT_WS : wsParam;
            const handle = connectLive(model.plugin, url);
            return () => handle.close();
        }
        const mvsjParam = params.get('mvsj');
        if (mvsjParam !== null) {
            const url = mvsjParam === '' ? 'volume.mvsj' : mvsjParam;
            loadMVSFromUrl(model.plugin, url, 'mvsj');
            return;
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
