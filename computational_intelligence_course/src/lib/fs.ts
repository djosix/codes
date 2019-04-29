const fs = eval('require("fs")');

export function readTextFile(path: string): string {
  return fs.readFileSync(path, 'utf8');
}

export function createDirectory(path: string) {
  fs.mkdirSync(path);
}

export function writeTextFile(path: string, content: string) {
  fs.writeFileSync(path, content, 'utf8');
}
