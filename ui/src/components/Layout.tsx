import { ReactNode, useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { Moon, Sun, Home, Upload, Network, MessageSquare, Database, Settings, Terminal } from "lucide-react";
import { Button } from "./ui/button";
import { cn } from "@/lib/utils";

interface LayoutProps {
  children: ReactNode;
}

export const Layout = ({ children }: LayoutProps) => {
  const [isDark, setIsDark] = useState(() => {
    // Initialize from localStorage or default to false
    const saved = localStorage.getItem("theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    return saved === "dark" || (!saved && prefersDark);
  });
  const location = useLocation();

  // Apply theme on mount and when it changes
  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [isDark]);

  const toggleTheme = () => {
    setIsDark(!isDark);
  };

  const navItems = [
    { path: "/", label: "Dashboard", shortLabel: "Dashboard", icon: Home },
    { path: "/data-sources", label: "Data Sources", shortLabel: "Sources", icon: Upload },
    { path: "/graph", label: "Graph", shortLabel: "Graph", icon: Network },
    { path: "/vectors", label: "Vectors", shortLabel: "Vectors", icon: Database },
    { path: "/chat", label: "Chat", shortLabel: "Chat", icon: MessageSquare },
    { path: "/logs", label: "Logs", shortLabel: "Logs", icon: Terminal },
  ];

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <Link to="/" className="flex items-center gap-2">
              <img src="/logo.png" alt="MIRAGE Logo" className="w-10 h-10 rounded-lg object-contain" />
              <div>
                <h1 className="text-xl font-bold">MIRAGE</h1>
                <p className="text-xs text-muted-foreground">Multilingual Info Retrieval</p>
              </div>
            </Link>

            <nav className="hidden md:flex items-center gap-2">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.path;
                return (
                  <Link key={item.path} to={item.path}>
                    <Button
                      variant={isActive ? "secondary" : "ghost"}
                      size="sm"
                      className={cn("gap-2", isActive && "bg-secondary")}
                    >
                      <Icon className="w-4 h-4" />
                      {item.label}
                    </Button>
                  </Link>
                );
              })}
            </nav>

            <div className="flex items-center gap-2">
              <Link to="/settings">
                <Button
                  variant={location.pathname === "/settings" ? "secondary" : "outline"}
                  size="icon"
                >
                  <Settings className="w-5 h-5" />
                </Button>
              </Link>
              <Button variant="outline" size="icon" onClick={toggleTheme}>
                {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">{children}</main>

      {/* Mobile Navigation */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 bg-card border-t border-border backdrop-blur-sm">
        <div className="flex items-center justify-around p-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            return (
              <Link key={item.path} to={item.path} className="flex-1">
                <Button
                  variant={isActive ? "secondary" : "ghost"}
                  size="sm"
                  className={cn("w-full flex flex-col gap-1 h-auto py-2", isActive && "bg-secondary")}
                >
                  <Icon className="w-5 h-5" />
                  <span className="text-xs">{item.shortLabel}</span>
                </Button>
              </Link>
            );
          })}
        </div>
      </nav>
    </div>
  );
};
