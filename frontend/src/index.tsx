import { useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { useCreatePluginViewModel } from 'molstar/lib/extensions/plugin/hooks/use-view-model';
import { PluginCanvas } from 'molstar/lib/extensions/plugin/react';
import { loadPdb } from 'molstar/lib/extensions/plugin/loaders';

function App() {
    const model = useCreatePluginViewModel();

    useEffect(() => {
        loadPdb(model.plugin, '1tqn');
    }, [model]);

    return (
        <div style={{ position: 'absolute', inset: 0 }}>
            <PluginCanvas model={model} />
        </div>
    );
}

createRoot(document.getElementById('app')!).render(<App />);
