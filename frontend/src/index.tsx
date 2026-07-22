import { useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { MolViewSpecBehavior } from 'molstar/lib/extensions/mvs/behavior';
import { useCreatePluginViewModel } from 'molstar/lib/extensions/plugin/hooks/use-view-model';
import { PluginCanvas } from 'molstar/lib/extensions/plugin/react';
import { loadMVSFromUrl, loadPdb } from 'molstar/lib/extensions/plugin/loaders';
import { PluginSpec } from 'molstar/lib/mol-plugin/spec';
import { Interactions } from 'molstar/lib/mol-plugin/behavior/dynamic/custom-props/computed/interactions';
import { connectLive } from './live';

const DEFAULT_WS = 'ws://127.0.0.1:8787';

function App() {
    const model = useCreatePluginViewModel({
        spec: (s) => {
            s.behaviors.push(MolViewSpecBehavior);
            // Registers the 'interactions' representation type and its computed
            // custom property, so `set_interactions` from Python has something
            // to add. Not in the minimal default spec we start from.
            s.behaviors.push(PluginSpec.Behavior(Interactions));
            // No XYZ axes indicator — off from the start (in the default props, so it never
            // flashes on a fresh page), and there is no toggle for it.
            s.canvas3d = { ...(s.canvas3d || {}), camera: { helper: { axes: { name: 'off', params: {} } } } } as any;
            return s;
        },
    });

    useEffect(() => {
        const params = new URLSearchParams(window.location.search);
        // `?ws=ws://host:port` streams live coordinates from a Python LiveSession
        // and can also drive per-volume color/opacity commands. Several may be given
        // comma-separated — each becomes an independent structure in the one plugin
        // (multi-model). `?mvsj=path/to/scene.mvsj` loads a static MVSJ scene (e.g. a
        // volume demo). With no `ws`/`mvsj` we fall back to a static PDB so the page
        // is never blank.
        const mvsjParam = params.get('mvsj');
        const wsParam = params.get('ws');

        let liveHandles: ReturnType<typeof connectLive>[] = [];

        const setup = async () => {
            if (mvsjParam !== null) {
                const url = mvsjParam === '' ? 'volume.mvsj' : mvsjParam;
                await loadMVSFromUrl(model.plugin, url, 'mvsj');
            }
            if (wsParam !== null) {
                const urls = wsParam.split(',').map((u) => u.trim()).filter(Boolean);
                liveHandles = (urls.length ? urls : [DEFAULT_WS]).map((url) => connectLive(model.plugin, url));
            }
            if (mvsjParam === null && wsParam === null) {
                loadPdb(model.plugin, '1tqn');
            }
        };

        // Expose the plugin for headless render/verification harnesses (opt-in via ?debug).
        if (params.has('debug')) (window as any).__pxviewer_plugin = model.plugin;
        setup();

        return () => {
            liveHandles.forEach((h) => h.close());
        };
    }, [model]);

    return (
        <div style={{ position: 'absolute', inset: 0 }}>
            <PluginCanvas model={model} />
        </div>
    );
}

createRoot(document.getElementById('app')!).render(<App />);
