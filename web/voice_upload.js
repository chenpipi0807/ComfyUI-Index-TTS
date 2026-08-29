import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const PLACEHOLDER = "(暂无音色，可点击上传 .pt)";

function setVoiceOptions(widget, files, selected = null) {
    if (!widget) return;
    const values = files.length ? files : [PLACEHOLDER];
    widget.options.values = values;

    if (selected && values.includes(selected)) {
        widget.value = selected;
    } else if (!values.includes(widget.value)) {
        widget.value = values[0];
    }
}

async function refreshVoiceFiles(widget, node, selected = null) {
    const response = await api.fetchApi("/indextts25/voice_files");
    if (!response.ok) {
        throw new Error(await response.text());
    }
    const payload = await response.json();
    setVoiceOptions(widget, payload.files || [], selected);
    node.setDirtyCanvas(true, true);
}

async function uploadVoiceFile(widget, node) {
    const picker = document.createElement("input");
    picker.type = "file";
    picker.accept = ".pt";
    picker.style.display = "none";
    document.body.appendChild(picker);

    picker.onchange = async () => {
        try {
            const file = picker.files?.[0];
            if (!file) return;
            if (!file.name.toLowerCase().endsWith(".pt")) {
                throw new Error("请选择 .pt 音色文件");
            }

            const form = new FormData();
            form.append("file", file, file.name);
            const response = await api.fetchApi("/indextts25/upload_voice", {
                method: "POST",
                body: form,
            });
            const payload = await response.json();
            if (!response.ok) {
                throw new Error(payload.error || "上传失败");
            }

            await refreshVoiceFiles(widget, node, payload.name);
        } catch (error) {
            alert(`IndexTTS Voice: ${error.message || error}`);
        } finally {
            picker.remove();
        }
    };

    picker.oncancel = () => picker.remove();
    picker.click();
}

app.registerExtension({
    name: "ComfyUI.IndexTTS25.VoiceUpload",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "IndexTTS25LoadVoiceNode") return;

        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = original?.apply(this, arguments);
            const voiceWidget = this.widgets?.find((widget) => widget.name === "voice_file");

            this.addWidget("button", "上传 .pt", null, () => uploadVoiceFile(voiceWidget, this));
            this.addWidget("button", "刷新音色列表", null, async () => {
                try {
                    await refreshVoiceFiles(voiceWidget, this);
                } catch (error) {
                    alert(`IndexTTS Voice: ${error.message || error}`);
                }
            });

            return result;
        };
    },
});
