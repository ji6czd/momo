import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";
import { MomoWasm } from "../wasm/momors_wasm.js";

let momo: MomoWasm | undefined;

function getMomo(context: vscode.ExtensionContext): MomoWasm {
  if (!momo) {
    const modelPath = path.join(
      context.extensionPath,
      "resources",
      "model.mbm",
    );
    const modelBytes = fs.readFileSync(modelPath);
    momo = new MomoWasm(new Uint8Array(modelBytes));
  }
  return momo;
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

// predict_segmented の JSON から分かち書きかな文字列を組み立てる
function buildSegmentedString(text: string, jsonStr: string): string {
  const data = JSON.parse(jsonStr) as {
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

export function activate(context: vscode.ExtensionContext) {
  const disposableKana = vscode.commands.registerCommand(
    "momo-translator.to-kana",
    () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage("No active editor found.");
        return;
      }
      try {
        const currentLine = editor.selection.active.line;
        const lineText = editor.document.lineAt(currentLine).text;
        const kana = getMomo(context).predictKana(lineText);
        insertLineAfterCurrent(editor, currentLine, kana).then(
          undefined,
          (err: Error) => vscode.window.showErrorMessage(String(err)),
        );
      } catch (err) {
        vscode.window.showErrorMessage(String(err));
      }
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
      try {
        const currentLine = editor.selection.active.line;
        const lineText = editor.document.lineAt(currentLine).text;
        const braille = getMomo(context).predictBraille(lineText);
        insertLineAfterCurrent(editor, currentLine, braille).then(
          undefined,
          (err: Error) => vscode.window.showErrorMessage(String(err)),
        );
      } catch (err) {
        vscode.window.showErrorMessage(String(err));
      }
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
      try {
        const currentLine = editor.selection.active.line;
        const lineText = editor.document.lineAt(currentLine).text;
        const jsonStr = getMomo(context).predictSegmented(lineText);
        const result = buildSegmentedString(lineText, jsonStr);
        insertLineAfterCurrent(editor, currentLine, result).then(
          undefined,
          (err: Error) => vscode.window.showErrorMessage(String(err)),
        );
      } catch (err) {
        vscode.window.showErrorMessage(String(err));
      }
    },
  );

  context.subscriptions.push(disposableKana);
  context.subscriptions.push(disposableBraille);
  context.subscriptions.push(disposableSegmented);
}

export function deactivate() {
  momo?.free();
  momo = undefined;
}
