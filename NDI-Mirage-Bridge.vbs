Set WshShell = CreateObject("WScript.Shell")
' Resolve paths relative to this script
strDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
strPython = strDir & "\.venv\Scripts\pythonw.exe"
strScript = strDir & "\ndi_mirage_bridge_ui.py"
WshShell.CurrentDirectory = strDir
WshShell.Run """" & strPython & """ """ & strScript & """", 0, False
