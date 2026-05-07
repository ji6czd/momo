// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from 'vscode';

// momoを実行して、翻訳結果を取得する関数
async function translateToKana(text: string, braille: boolean): Promise<string> {
	const config = vscode.workspace.getConfiguration('momoTranslator');
	const momoCommand = config.get<string>('command') || 'momo';
	const model = config.get<string>('model') || 'basic_data.zip';
	const commandArgs = braille
		? ['predict', '--no-source', '--model', model, '--no-kana']
		: ['predict', '--segment', '--no-source', '--model', model];
	return new Promise((resolve, reject) => {
		const { spawn } = require('child_process');
		const env = { ...process.env, PYTHONIOENCODING: 'utf-8' };
		const child = spawn(momoCommand, commandArgs, { env });
		
		let stdout = '';
		let stderr = '';

		child.stdout.on('data', (data: Buffer) => {
			stdout += data.toString('utf-8');
		});

		child.stderr.on('data', (data: Buffer) => {
			stderr += data.toString('utf-8');
		});

		child.on('close', (code: number) => {
			if (code !== 0) {
				reject(`Error executing momo: exit code ${code}, stderr: ${stderr}`);
				return;
			}
			if (stderr) {
				reject(`Error from momo: ${stderr}`);
				return;
			}
			resolve(stdout.trim());
		});

		child.on('error', (error: Error) => {
			reject(`Error executing momo: ${error.message}`);
		});

		// 標準入力にテキストをUTF-8で送信
		child.stdin.write(text, 'utf-8');
		child.stdin.end();
	});
}

// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed
export function activate(context: vscode.ExtensionContext) {
	const disposableKana = vscode.commands.registerCommand('momo-translator.to-kana', () => {
		const editor = vscode.window.activeTextEditor;
		if (!editor) {
			vscode.window.showErrorMessage('No active editor found.');
			return;
		}
		// 現在編集中の行を取得
		const currentLine = editor.selection.active.line;
		const lineText = editor.document.lineAt(currentLine).text;
		
		translateToKana(lineText, false).then(translation => {
			editor.edit((editBuilder: vscode.TextEditorEdit) => {
				// 現在の行の直後に結果を挿入
				const nextLine = currentLine + 1;
				const insertPosition = new vscode.Position(nextLine, 0);
				editBuilder.insert(insertPosition, translation + '\n');
			});
		}
		).catch(error => {
			vscode.window.showErrorMessage(error);
		});
	});

	const disposableBraille = vscode.commands.registerCommand('momo-translator.to-braille', () => {
		const editor = vscode.window.activeTextEditor;
		if (!editor) {
			vscode.window.showErrorMessage('No active editor found.');
			return;
		}
		// 現在編集中の行を取得
		const currentLine = editor.selection.active.line;
		const lineText = editor.document.lineAt(currentLine).text;
		
		translateToKana(lineText, true).then(translation => {
			editor.edit((editBuilder: vscode.TextEditorEdit) => {
				// 現在の行の直後に結果を挿入
				const nextLine = currentLine + 1;
				const insertPosition = new vscode.Position(nextLine, 0);
				editBuilder.insert(insertPosition, translation + '\n');
			});
		}
		).catch(error => {
			vscode.window.showErrorMessage(error);
		});
	});

		context.subscriptions.push(disposableKana);
		context.subscriptions.push(disposableBraille);
	}

// This method is called when your extension is deactivated
export function deactivate() { }
