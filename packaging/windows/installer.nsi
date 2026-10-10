; pxviewer Windows installer — NSIS Modern UI, per-user install (no UAC).
;
; Build:  makensis /DVERSION=0.1.0 /DSTAGEDIR=<dir> /DBUILDPREFIX=<dir> installer.nsi
;
; STAGEDIR must contain:
;   env.tar.xz      the conda environment as ONE archive — NSIS cannot mmap a
;                   datablock of ~300k files / multi-GB payload, so the env is
;                   embedded compressed and extracted by the OS's own tar.exe
;                   at install time
;   prefix_fixup.py post-install path rewriter
;   pxviewer.ico    app icon
;   LICENSE
;
; The installer is unsigned: SmartScreen shows "Windows protected your PC" —
; click "More info" -> "Run anyway". Signing needs an EV/OV certificate, same
; story as the macOS dmg.

Unicode true
!include "MUI2.nsh"
!include "LogicLib.nsh"

!ifndef VERSION
  !define VERSION "0.0.0"
!endif
!ifndef STAGEDIR
  !define STAGEDIR "stage"
!endif
!ifndef BUILDPREFIX
  !define BUILDPREFIX "C:\pxviewer-build\env"
!endif

!ifndef OUTFILE
  !define OUTFILE "pxviewer-${VERSION}-windows-x86_64-setup.exe"
!endif

Name "pxviewer ${VERSION}"
OutFile "${OUTFILE}"
InstallDir "$LOCALAPPDATA\Programs\pxviewer"
RequestExecutionLevel user          ; per-user install, no admin prompt
; Not /SOLID: env.tar.xz is stored as-is (SetCompress off below), which a solid
; block would ignore. Recompressing the xz gained nothing (101% of input) and
; cost an extra full LZMA pass at both build and install time.
SetCompressor lzma

!define MUI_ICON "${STAGEDIR}\pxviewer.ico"
!define MUI_UNICON "${STAGEDIR}\pxviewer.ico"
!define MUI_ABORTWARNING

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "${STAGEDIR}\LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

Section "pxviewer" SecMain
  SetOutPath "$INSTDIR"
  File /r /x env.tar.xz "${STAGEDIR}\*.*"
  SetCompress off
  File "${STAGEDIR}\env.tar.xz"
  SetCompress auto

  ; Unpack the env archive with the OS's bsdtar (C:\Windows\System32\tar.exe,
  ; present since Windows 10 17063 — our floor is far above that).
  ; The installer is a 32-bit process, so $SYSDIR is redirected to SysWOW64,
  ; whose 32-bit tar fails on this archive ("Lzma library error: Corrupted
  ; input data"). Sysnative reaches the real 64-bit System32 from WOW64.
  DetailPrint "Extracting environment (a few minutes)..."
  nsExec::ExecToLog '"$WINDIR\Sysnative\tar.exe" -xf "$INSTDIR\env.tar.xz" -C "$INSTDIR"'
  Pop $0
  ${If} $0 != 0
    Abort "Failed to extract the bundled environment (tar.exe exit $0)"
  ${EndIf}
  Delete "$INSTDIR\env.tar.xz"

  ; The env was built at ${BUILDPREFIX}; rewrite embedded paths to $INSTDIR.
  DetailPrint "Finalizing environment (rewriting install paths)..."
  nsExec::ExecToLog '"$INSTDIR\env\python.exe" "$INSTDIR\prefix_fixup.py" "${BUILDPREFIX}" "$INSTDIR\env"'

  ; Start Menu: launch via pythonw so no console window appears. Working dir is
  ; whatever SetOutPath last set — give the app its install dir.
  SetOutPath "$INSTDIR"
  CreateDirectory "$SMPROGRAMS\pxviewer"
  CreateShortcut "$SMPROGRAMS\pxviewer\pxviewer.lnk" \
    '"$INSTDIR\env\pythonw.exe"' "-m pxviewer desktop" "$INSTDIR\pxviewer.ico"
  ; console variant for debugging — opens a terminal with stdout/stderr
  CreateShortcut "$SMPROGRAMS\pxviewer\pxviewer (console).lnk" \
    '"$INSTDIR\env\python.exe"' "-m pxviewer desktop" "$INSTDIR\pxviewer.ico"

  ; Add/Remove Programs entry
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\pxviewer" \
    "DisplayName" "pxviewer"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\pxviewer" \
    "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\pxviewer" \
    "Publisher" "cschlick"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\pxviewer" \
    "DisplayIcon" "$INSTDIR\pxviewer.ico"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\pxviewer" \
    "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\pxviewer" \
    "InstallLocation" "$INSTDIR"
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\pxviewer" \
    "NoModify" 1
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\pxviewer" \
    "NoRepair" 1
SectionEnd

Section "Uninstall"
  ; RMDir /r removes the env's ~300k files one by one, printing each — over 10
  ; minutes. rd /s /q does the same in a fraction of that. Guarded on a file
  ; only our env has, so a broken $INSTDIR can never point rd at something else.
  ${If} ${FileExists} "$INSTDIR\env\python.exe"
    DetailPrint "Removing environment (a minute or two)..."
    nsExec::Exec '"$SYSDIR\cmd.exe" /c rd /s /q "$INSTDIR\env"'
    Pop $0
  ${EndIf}
  RMDir /r "$INSTDIR"
  RMDir /r "$SMPROGRAMS\pxviewer"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\pxviewer"
SectionEnd
