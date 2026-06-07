// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as cp from "child_process";
import * as vscode from "vscode";

let momoServerProcess: cp.ChildProcess | null = null;

// サーバーのURLを設定から取得する
function getServerUrl(): string {
  const config = vscode.workspace.getConfiguration("momoTranslator");
  return (config.get<string>("serverUrl") || "http://127.0.0.1:8765").replace(
    /\/$/,
    "",
  );
}

// momo-server が起動していなければ自動起動し、応答を待つ
async function ensureServerRunning(): Promise<void> {
  const baseUrl = getServerUrl();
  try {
    await fetch(`${baseUrl}/health`, { signal: AbortSignal.timeout(1000) });
    return;
  } catch {
    // 応答なし → 起動を試みる
  }
  if (!momoServerProcess || momoServerProcess.exitCode !== null) {
    momoServerProcess = cp.spawn("momo-server", [], { stdio: "ignore" });
  }
  const deadline = Date.now() + 10000;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 500));
    try {
      await fetch(`${baseUrl}/health`, { signal: AbortSignal.timeout(1000) });
      return;
    } catch {
      // まだ未応答
    }
  }
  throw new Error("momo-server の起動がタイムアウトしました (10秒)。");
}

// momo-server の /predict/kana エンドポイントを呼び出し、仮名文字列を返す
async function translateToKana(text: string): Promise<string> {
  await ensureServerRunning();
  const url = `${getServerUrl()}/predict/kana`;
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
  } catch (err) {
    throw new Error(
      `momo-server に接続できません (${url})。サーバーが起動しているか確認してください。\n${err}`,
    );
  }
  if (!response.ok) {
    const detail = await response.text().catch(() => `HTTP ${response.status}`);
    throw new Error(`momo-server エラー (${response.status}): ${detail}`);
  }
  const data = (await response.json()) as { kana_text: string };
  return data.kana_text;
}

// momo-server の /predict/predict エンドポイントを呼び出し、点字文字列を返す
async function translateToBraille(text: string): Promise<string> {
  await ensureServerRunning();
  const url = `${getServerUrl()}/predict/predict`;
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
  } catch (err) {
    throw new Error(
      `momo-server に接続できません (${url})。サーバーが起動しているか確認してください。\n${err}`,
    );
  }
  if (!response.ok) {
    const detail = await response.text().catch(() => `HTTP ${response.status}`);
    throw new Error(`momo-server エラー (${response.status}): ${detail}`);
  }
  const data = (await response.json()) as { braille: string };
  return data.braille;
}

// momo-server の /predict/predict エンドポイントを呼び出し、分割された仮名文字列を返す
async function translateToSegmentedKana(text: string): Promise<string> {
  await ensureServerRunning();
  const url = `${getServerUrl()}/predict/predict`;
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
  } catch (err) {
    throw new Error(
      `momo-server に接続できません (${url})。サーバーが起動しているか確認してください。\n${err}`,
    );
  }
  if (!response.ok) {
    const detail = await response.text().catch(() => `HTTP ${response.status}`);
    throw new Error(`momo-server エラー (${response.status}): ${detail}`);
  }
  const data = (await response.json()) as {
    braille: string;
    kana: string;
    kana_to_src_index: number[];
    src_to_kana_index: number[][];
  };
  const segments: string[] = [];
  for (let i = 0; i < text.length; i++) {
    const kanaPositions = data.src_to_kana_index[i];
    if (!kanaPositions || kanaPositions.length === 0) {
      continue;
    }
    let current = "";
    for (const p of kanaPositions) {
      const ch = data.kana[p];
      if (ch === " ") {
        if (current) {
          segments.push(current);
          current = "";
        }
        segments.push(" ");
      } else {
        current += ch;
      }
    }
    if (current) {
      segments.push(current);
    }
  }
  return data.braille + "\n" + segments.join("/");
}

// 現在行の直後に1行挿入するヘルパー
function insertLineAfterCurrent(
  editor: vscode.TextEditor,
  line: number,
  text: string,
): Thenable<boolean> {
  return editor.edit((editBuilder: vscode.TextEditorEdit) => {
    const insertPosition = new vscode.Position(line + 1, 0);
    editBuilder.insert(insertPosition, text + "\n");
  });
}

// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed
export function activate(context: vscode.ExtensionContext) {
  const disposableKana = vscode.commands.registerCommand(
    "momo-translator.to-kana",
    () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage("No active editor found.");
        return;
      }
      const currentLine = editor.selection.active.line;
      const lineText = editor.document.lineAt(currentLine).text;

      translateToKana(lineText)
        .then((translation) =>
          insertLineAfterCurrent(editor, currentLine, translation),
        )
        .catch((error: Error) => {
          vscode.window.showErrorMessage(String(error.message ?? error));
        });
    },
  );

  const disposableBraille = vscode.commands.registerCommand(
    "momo-translator.to-braille",
    () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage("No active editor found.");
        return;
      }
      const currentLine = editor.selection.active.line;
      const lineText = editor.document.lineAt(currentLine).text;

      translateToBraille(lineText)
        .then((translation) =>
          insertLineAfterCurrent(editor, currentLine, translation),
        )
        .catch((error: Error) => {
          vscode.window.showErrorMessage(String(error.message ?? error));
        });
    },
  );

  const disposableSegmented = vscode.commands.registerCommand(
    "momo-translator.to-segmented",
    () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage("No active editor found.");
        return;
      }
      const currentLine = editor.selection.active.line;
      const lineText = editor.document.lineAt(currentLine).text;

      translateToSegmentedKana(lineText)
        .then((translation) =>
          insertLineAfterCurrent(editor, currentLine, translation),
        )
        .catch((error: Error) => {
          vscode.window.showErrorMessage(String(error.message ?? error));
        });
    },
  );

  context.subscriptions.push(disposableKana);
  context.subscriptions.push(disposableBraille);
  context.subscriptions.push(disposableSegmented);
}

// This method is called when your extension is deactivated
export function deactivate() {
  momoServerProcess?.kill();
}
