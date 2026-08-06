import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";
import {
  BrailleTranslator,
  Predictor,
  type PredictionResult,
} from "../wasm/momors_wasm.js";

let predictor: Predictor | undefined;
let translator: BrailleTranslator | undefined;

function getPredictor(context: vscode.ExtensionContext): Predictor {
  if (!predictor) {
    const modelPath = path.join(
      context.extensionPath,
      "resources",
      "model.mbm",
    );
    const modelBytes = fs.readFileSync(modelPath);
    predictor = new Predictor(new Uint8Array(modelBytes));
  }
  return predictor;
}

function getBrailleTranslator(): BrailleTranslator {
  if (!translator) {
    translator = new BrailleTranslator();
  }
  return translator;
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

// PredictionResult.sourceToKana から分かち書きかな文字列を組み立てる
function buildSegmentedString(
  text: string,
  braille: string,
  pred: PredictionResult,
): string {
  const kana = pred.kana;
  const sourceToKana = pred.sourceToKana;
  const segments: string[] = [];
  for (let i = 0; i < text.length; i++) {
    const kanaPositions: Iterable<number> | undefined = sourceToKana[i];
    if (!kanaPositions) {
      continue;
    }
    let current = "";
    for (const p of kanaPositions) {
      const ch = kana[p];
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
  return braille + "\n" + segments.join("/");
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
        const kana = getPredictor(context).predict(lineText).kana;
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
        const kana = getPredictor(context).predict(lineText).kana;
        const braille = getBrailleTranslator().translate(kana).braille;
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
        const pred = getPredictor(context).predict(lineText);
        const braille = getBrailleTranslator().translate(pred.kana).braille;
        const result = buildSegmentedString(lineText, braille, pred);
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
  predictor?.free();
  predictor = undefined;
  translator?.free();
  translator = undefined;
}
