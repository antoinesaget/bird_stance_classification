// ==UserScript==
// @name         BirdSys Label Studio Bulk Apply
// @namespace    https://birdsys.local
// @version      0.1.0
// @description  Adds an "Apply to selected birds" panel in Label Studio.
// @match        http://localhost:8080/*
// @match        http://127.0.0.1:8080/*
// @grant        none
// ==/UserScript==

(function () {
  "use strict";

  const INSTALL_KEY = "__birdsysBulkApplyInstalled";
  const PANEL_ID = "birdsys-bulk-apply-panel";

  if (window[INSTALL_KEY]) {
    return;
  }
  window[INSTALL_KEY] = true;

  const behaviorOptions = [
    { value: "flying", hotkey: "e" },
    { value: "foraging", hotkey: "r" },
    { value: "resting", hotkey: "t" },
    { value: "backresting", hotkey: "y" },
    { value: "preening", hotkey: "u" },
    { value: "display", hotkey: "i" },
    { value: "unsure", hotkey: "o" },
  ];

  const substrateOptions = [
    { value: "ground", hotkey: "g" },
    { value: "water", hotkey: "w" },
    { value: "air", hotkey: "v" },
    { value: "unsure", hotkey: "b" },
  ];

  const legsOptions = [
    { value: "skip", hotkey: "" },
    { value: "one", hotkey: "1" },
    { value: "two", hotkey: "2" },
    { value: "unsure", hotkey: "3" },
  ];

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function isLabelStudioAnnotationPage() {
    return /\/projects\/\d+\/data/.test(window.location.pathname);
  }

  function getSelectedBirdRows() {
    const selectors = [".ant-tree-treenode-selected", '[aria-selected="true"]'];
    const seen = new Set();
    const rows = [];

    for (const selector of selectors) {
      for (const el of document.querySelectorAll(selector)) {
        const row = el.closest(".ant-tree-treenode, [role='treeitem']") || el;
        if (!row || seen.has(row)) {
          continue;
        }
        seen.add(row);
        const text = (row.textContent || "").toLowerCase();
        if (text.includes("bird")) {
          rows.push(row);
        }
      }
    }

    return rows;
  }

  function dispatchKey(key) {
    if (!key) {
      return;
    }

    const keyCode = key.toUpperCase().charCodeAt(0);
    const events = ["keydown", "keyup"];

    for (const type of events) {
      document.dispatchEvent(
        new KeyboardEvent(type, {
          key,
          code: key.length === 1 && /\d/.test(key) ? `Digit${key}` : `Key${key.toUpperCase()}`,
          keyCode,
          which: keyCode,
          bubbles: true,
          cancelable: true,
        }),
      );
    }
  }

  async function applyAttributes({ behaviorKey, substrateKey, legsKey, statusEl }) {
    const selectedRows = getSelectedBirdRows();
    if (!selectedRows.length) {
      statusEl.textContent = "No selected birds. Ctrl/Cmd+click regions in the right sidebar first.";
      return;
    }

    statusEl.textContent = `Applying to ${selectedRows.length} birds...`;

    for (let idx = 0; idx < selectedRows.length; idx += 1) {
      const row = selectedRows[idx];
      row.scrollIntoView({ block: "nearest" });
      row.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
      await sleep(90);

      dispatchKey(behaviorKey);
      await sleep(60);
      dispatchKey(substrateKey);
      await sleep(60);

      if (legsKey) {
        dispatchKey(legsKey);
        await sleep(60);
      }

      statusEl.textContent = `Applied ${idx + 1}/${selectedRows.length} birds...`;
    }

    statusEl.textContent = `Done. Updated ${selectedRows.length} birds.`;
  }

  function createSelect(id, options, defaultValue) {
    const select = document.createElement("select");
    select.id = id;
    select.style.width = "100%";
    select.style.padding = "6px 8px";
    select.style.marginTop = "4px";
    select.style.border = "1px solid #c9c9c9";
    select.style.borderRadius = "6px";
    select.style.fontSize = "13px";

    for (const option of options) {
      const el = document.createElement("option");
      el.value = option.hotkey;
      el.textContent = `${option.value}${option.hotkey ? ` [${option.hotkey}]` : ""}`;
      if (option.value === defaultValue) {
        el.selected = true;
      }
      select.appendChild(el);
    }

    return select;
  }

  function createPanel() {
    if (document.getElementById(PANEL_ID) || !isLabelStudioAnnotationPage()) {
      return;
    }

    const panel = document.createElement("div");
    panel.id = PANEL_ID;
    panel.style.position = "fixed";
    panel.style.top = "96px";
    panel.style.left = "16px";
    panel.style.width = "280px";
    panel.style.zIndex = "9999";
    panel.style.background = "rgba(255,255,255,0.96)";
    panel.style.border = "1px solid #dadada";
    panel.style.borderRadius = "10px";
    panel.style.boxShadow = "0 6px 20px rgba(0,0,0,0.15)";
    panel.style.padding = "12px";
    panel.style.fontFamily = "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial";
    panel.style.fontSize = "13px";
    panel.style.color = "#222";

    const title = document.createElement("div");
    title.textContent = "Bird Bulk Apply";
    title.style.fontSize = "14px";
    title.style.fontWeight = "700";
    title.style.marginBottom = "10px";
    panel.appendChild(title);

    const behaviorLabel = document.createElement("label");
    behaviorLabel.setAttribute("for", "birdsys-behavior");
    behaviorLabel.textContent = "Behavior";
    panel.appendChild(behaviorLabel);
    const behaviorSelect = createSelect("birdsys-behavior", behaviorOptions, "flying");
    panel.appendChild(behaviorSelect);

    const substrateLabel = document.createElement("label");
    substrateLabel.setAttribute("for", "birdsys-substrate");
    substrateLabel.style.display = "block";
    substrateLabel.style.marginTop = "8px";
    substrateLabel.textContent = "Substrate";
    panel.appendChild(substrateLabel);
    const substrateSelect = createSelect("birdsys-substrate", substrateOptions, "air");
    panel.appendChild(substrateSelect);

    const legsLabel = document.createElement("label");
    legsLabel.setAttribute("for", "birdsys-legs");
    legsLabel.style.display = "block";
    legsLabel.style.marginTop = "8px";
    legsLabel.textContent = "Legs";
    panel.appendChild(legsLabel);
    const legsSelect = createSelect("birdsys-legs", legsOptions, "skip");
    panel.appendChild(legsSelect);

    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "Apply To Selected Birds";
    button.style.width = "100%";
    button.style.marginTop = "12px";
    button.style.padding = "8px 10px";
    button.style.background = "#1f8f4f";
    button.style.color = "#fff";
    button.style.border = "none";
    button.style.borderRadius = "6px";
    button.style.cursor = "pointer";
    button.style.fontWeight = "600";
    panel.appendChild(button);

    const hint = document.createElement("div");
    hint.textContent = "Select birds in the right sidebar with Ctrl/Cmd+click, then apply.";
    hint.style.marginTop = "8px";
    hint.style.color = "#555";
    panel.appendChild(hint);

    const status = document.createElement("div");
    status.style.marginTop = "8px";
    status.style.minHeight = "16px";
    status.style.color = "#1d5f3a";
    panel.appendChild(status);

    button.addEventListener("click", async () => {
      button.disabled = true;
      button.style.opacity = "0.7";
      try {
        await applyAttributes({
          behaviorKey: behaviorSelect.value,
          substrateKey: substrateSelect.value,
          legsKey: legsSelect.value,
          statusEl: status,
        });
      } catch (error) {
        status.textContent = `Error: ${error instanceof Error ? error.message : String(error)}`;
      } finally {
        button.disabled = false;
        button.style.opacity = "1";
      }
    });

    document.body.appendChild(panel);
  }

  function boot() {
    createPanel();
  }

  boot();
  new MutationObserver(() => {
    if (!document.getElementById(PANEL_ID)) {
      createPanel();
    }
  }).observe(document.documentElement, { childList: true, subtree: true });
})();
