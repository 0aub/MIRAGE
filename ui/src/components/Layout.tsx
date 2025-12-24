import { ReactNode, useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import {
  Moon, Sun, Home, Upload, Network, MessageSquare,
  Database, Settings, Terminal, BarChart3, Menu, X, Minimize2, Target
} from "lucide-react";
import { Button } from "./ui/button";
import { cn } from "@/lib/utils";
import { Separator } from "./ui/separator";

interface LayoutProps {
  children: ReactNode;
}

export const Layout = ({ children }: LayoutProps) => {
  const [isDark, setIsDark] = useState(() => {
    const saved = localStorage.getItem("theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    return saved === "dark" || (!saved && prefersDark);
  });
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [isDark]);

  // Close sidebar on route change (mobile)
  useEffect(() => {
    setSidebarOpen(false);
  }, [location.pathname]);

  const toggleTheme = () => setIsDark(!isDark);

  const navItems = [
    { path: "/", label: "Dashboard", icon: Home },
    { path: "/data-sources", label: "Data Sources", icon: Upload },
    { path: "/graph", label: "Graph", icon: Network },
    { path: "/vectors", label: "Vectors", icon: Database },
    { path: "/chat", label: "Chat", icon: MessageSquare },
    { path: "/evaluations", label: "Evaluations", icon: BarChart3 },
    { path: "/logs", label: "Logs", icon: Terminal },
  ];

  const SidebarContent = () => (
    <div className="flex flex-col h-full">
      {/* Logo Section */}
      <div className="p-6 pb-4">
        <Link to="/" className="flex flex-col items-center gap-3">
          <img
            src="/logo.png"
            alt="MIRAGE Logo"
            className="w-16 h-16 rounded-xl object-contain shadow-lg"
          />
          <div className="text-center">
            <h1 className="text-2xl font-bold tracking-tight">MIRAGE</h1>
            <p className="text-xs text-muted-foreground">Multilingual Info Retrieval</p>
          </div>
        </Link>
      </div>

      <Separator className="mx-4" />

      {/* Navigation Items */}
      <nav className="flex-1 p-4 space-y-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = location.pathname === item.path;
          return (
            <Link key={item.path} to={item.path}>
              <Button
                variant={isActive ? "secondary" : "ghost"}
                className={cn(
                  "w-full justify-start gap-3 h-11",
                  isActive && "bg-secondary font-medium"
                )}
              >
                <Icon className="w-5 h-5" />
                {item.label}
              </Button>
            </Link>
          );
        })}
      </nav>

      <Separator className="mx-4" />

      {/* Bottom Section - Settings & Theme */}
      <div className="p-4 space-y-1">
        <Link to="/settings">
          <Button
            variant={location.pathname === "/settings" ? "secondary" : "ghost"}
            className={cn(
              "w-full justify-start gap-3 h-11",
              location.pathname === "/settings" && "bg-secondary font-medium"
            )}
          >
            <Settings className="w-5 h-5" />
            Settings
          </Button>
        </Link>

        <Button
          variant="ghost"
          className="w-full justify-start gap-3 h-11"
          onClick={toggleTheme}
        >
          {isDark ? (
            <>
              <Sun className="w-5 h-5" />
              Light Mode
            </>
          ) : (
            <>
              <Moon className="w-5 h-5" />
              Dark Mode
            </>
          )}
        </Button>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-background flex">
      {/* Desktop Sidebar */}
      <aside className="hidden md:flex w-64 flex-col border-r border-border bg-card/50 fixed h-screen">
        <SidebarContent />
      </aside>

      {/* Mobile Header */}
      <header className="md:hidden fixed top-0 left-0 right-0 h-14 border-b border-border bg-card/95 backdrop-blur-sm z-50 flex items-center px-4">
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setSidebarOpen(true)}
        >
          <Menu className="w-5 h-5" />
        </Button>
        <div className="flex items-center gap-2 mx-auto">
          <img src="/logo.png" alt="MIRAGE" className="w-8 h-8 rounded-lg" />
          <span className="font-semibold">MIRAGE</span>
        </div>
        <Button variant="ghost" size="icon" onClick={toggleTheme}>
          {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
        </Button>
      </header>

      {/* Mobile Sidebar Overlay */}
      {sidebarOpen && (
        <div
          className="md:hidden fixed inset-0 bg-black/50 z-50"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Mobile Sidebar */}
      <aside
        className={cn(
          "md:hidden fixed top-0 left-0 h-screen w-64 bg-card border-r border-border z-50 transform transition-transform duration-200",
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <div className="absolute top-4 right-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setSidebarOpen(false)}
          >
            <X className="w-5 h-5" />
          </Button>
        </div>
        <SidebarContent />
      </aside>

      {/* Main Content */}
      <main className="flex-1 md:ml-64">
        <div className="container mx-auto px-4 py-6 md:py-8 mt-14 md:mt-0">
          {children}
        </div>
      </main>
    </div>
  );
};
